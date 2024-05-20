import numpy as np
from anomalies._base import Anomaly
from anomalies.increasing import Disturbance, GWN, Constant
from scipy import optimize
from scipy import linalg
from scipy import interpolate

# class Mixing(Disturbance): 
#     def _invert_deviation(self):
#         return np.sqrt( 1 - (1 - self.delta / 2)**2 )
    
#     def distort(self, Xok):
#         if Xok.ndim == 1:
#             N = 1
#         else:
#             N = len(Xok)
#         disturbance = self.generate(N)
#         return disturbance + np.sqrt(1-self.a**2) * Xok 
    
# class MixingGWN(GWN):
#     pass
    
# class MixingConstant(Constant):
#     pass

class Mixing(Disturbance): 
    def __init__(self, delta):
        self.delta = delta
    
    def fit(self, Xok):
        super().fit(Xok)
        N = Xok.shape[-2]
        self.Sok = 1/(N-1) * Xok.T @ Xok
        self.energy_per_component = np.trace(self.Sok) / self.n # for a standardized normal data is 1
        self.d_max = 2 * self.energy_per_component
        return self
    
    def _invert_deviation(self):
        return np.sqrt( 1 - (1 - self.delta / self.d_max)**2 )
    
        # alphas = 1 - ((2 - deltas)/2)**2
    
    def distort(self, Xok):
        if Xok.ndim == 1:
            N = 1
        else:
            N = len(Xok)
        self.a = self._invert_deviation()
        disturbance = self.generate(N) # disturbance with a unit energy per component
        disturbance = np.sqrt(self.energy_per_component) * disturbance # ensure energy per component equal to that of Xok
        return self.a * disturbance + np.sqrt(1-self.a**2) * Xok 
    
    
class MixingGWN(Mixing):
    def fit(self, Xok):
        super().fit(Xok) 
        self.gwn = GWN(1).fit(Xok)
        return self
        
    def generate(self, N):
        return self.gwn.generate(N)
    
    
class MixingConstant(Mixing):
    def fit(self, Xok):
        super().fit(Xok) 
        self.constant = Constant(1).fit(Xok)
        return self
        
    def generate(self, N):
        return self.constant.generate(N)
    

class TimeWarping(Anomaly):
    
    def __init__(self, delta, fs=1):
        super().__init__(delta)
        self.alpha = None
        self.fs = fs
    
    def _bspline3(self, x):
        return 1/6 * (np.max([0, 2-np.abs(x)])**3) \
    - (2/3)*np.max([0, 1-np.abs(x)])**3

    def _filt_bspline3(self, k):
        a = -2 + np.sqrt(3)
        return (-6*a)*(a**np.abs(k))/(1-a**2)
                                      
    def _card_spline3(self, x, m=10):
        return np.sum([
            self._filt_bspline3(k) * self._bspline3(x-k)
            for k in np.arange(-m, m)]
        )
    
    def _warpfun(self, tt, alpha):
        return (1-alpha)*tt
    
    def _deviation(self, alpha):
    
        # deviation is computed through sinc interpolation
        W_sinc = np.concatenate(
            [
                np.sinc(self._warpfun(i, alpha)-np.arange(self.n))
                for i in range(self.n)
            ]
        ).reshape(self.n, self.n)
        
        # compute scaling parameters
        scale1 = np.sqrt(np.trace(W_sinc.T @ W_sinc @ self.Sok) / self.n)
        scale2 = np.sqrt(np.trace(self.Sok) / self.n)
        return 2*scale2**2 - 2/self.n * np.trace(W_sinc.T @ self.Sok) / scale1 * scale2
    
#     def _invert_deviation_old(self):
#         # find and approximatin to alpha
#         alphas = np.linspace(0.5, 1, 100)
#         deviations = np.array(
#             [
#                 self._deviation(alpha) for alpha in alphas],
#             dtype=np.float64
#         )
#         delta_alpha = lambda alpha : self._deviation(alpha) - self.delta
#         alpha0 = alphas[np.argmin(np.abs(deviations - self.delta))]
        
#         # invert deviation starting from a good approximation
#         alpha = optimize.fsolve(delta_alpha, alpha0)[0]
#         return alpha
    
    def _invert_deviation(self):
        # invert deviation in the range of alpha [0, 1]
        delta_alpha = lambda alpha : self._deviation(alpha) - self.delta
        alpha = optimize.brentq(delta_alpha, a=0, b=1)
        return alpha
        
    def fit(self, Xok):
        super().fit(Xok)
        
        N = Xok.shape[-2]
        self.Sok = 1/(N-1) * Xok.T @ Xok
    
        # alpha corresponding to deviation level delta
        self.alpha = self._invert_deviation()
        
        # warping matrix
        W = np.zeros((self.n,self.n))
        for j in range(self.n):
            for i in range(self.n):
                W[j, i] = self._card_spline3(
                    self.fs*self._warpfun(j/self.fs, self.alpha) - i
                )
        self.scaling = np.sqrt(np.trace(self.Sok) \
                               / np.trace(W.T @ W @ self.Sok))
        
        return self
        
    def distort(self, Xok):
        ndim = 2
        if Xok.ndim == 1:
            ndim = 1
            Xok = Xok.reshape(1, self.n)
        Xko = np.zeros(Xok.shape)
        for i in range(len(Xok)):
            # interpolate with spline
            spline = interpolate.splrep(np.arange(self.n), Xok[i], s=0, k=3)
            tt = self._warpfun(np.arange(self.n)/self.fs, self.alpha)# warped time
            Xko[i] = interpolate.splev(tt, spline, der=0)
        # scale
        Xko = self.scaling * Xko
        if ndim==1:
            Xko = Xko.reshape(self.n)
            
        return Xko
    

class CovarianceAlterations(Anomaly):
    def __init__(self, delta):
        super().__init__(delta)
        self.theta = None
        self.d_max = None
        
    # def _invert_deviation(self):
    #     d_max = 2 * self.gamma
    #     theta = np.arccos(1-self.delta/d_max)
    #     return theta
    
    def _invert_deviation(self, d_max):
        theta = np.arccos(1-self.delta/d_max)
        return theta
    
    def r_theta(self, theta_):
        return np.array([
            [np.cos(theta_), -np.sin(theta_)],
            [np.sin(theta_), np.cos(theta_)]
        ])
        
    def fit(self, Xok):
        super().fit(Xok)
        # theta corresponding to deviation delta
        N = Xok.shape[-2]
        self.Sok = 1/(N-1) * Xok.T @ Xok
        eigvals, eigvecs = np.linalg.eigh(self.Sok)
        self.lok = eigvals[::-1]
        self.Uok = eigvecs[:, ::-1]
        
        self.energy_per_component = np.sum(self.lok) / self.n

        # # rotation matrix of angle theta acting on the principal subspace
        # self.theta = self._invert_deviation()
        # self.r_theta = lambda theta_: np.array([
        #     [np.cos(theta_), -np.sin(theta_)],
        #     [np.sin(theta_), np.cos(theta_)]
        # ])
        
        return self
    
    
class SpectralAlteration(CovarianceAlterations):
   
    def fit(self, Xok, SNRdB=None):
        super().fit(Xok)
        # infer the dimension of the principal subspace (PS) k
        singvalues = np.sqrt(self.lok)
        if SNRdB is None:
            threshold = 2.858 * np.median(singvalues)
        else:
            sigma = 10**(-SNRdB/20)
            threshold = 4/np.sqrt(3) * sigma * np.sqrt(self.n)
        
        dim_princ_sub = np.sum(singvalues > threshold)
        if dim_princ_sub%2:
            dim_princ_sub = dim_princ_sub+1
        self.dim_princ_sub = dim_princ_sub
        
        # principal subspace (PS) components
        self.Uoks = self.Uok[:, :self.dim_princ_sub]
        self.loks = self.lok[:self.dim_princ_sub]
        # noise subspace (NS) components
        self.Uokn = self.Uok[:, self.dim_princ_sub:]
        self.lokn = self.lok[self.dim_princ_sub:]
        # fraction of the signal energy contained in the PS
        self.gamma = np.sum(self.loks) / self.n
        
        # rotation matrix of angle theta acting on the principal subspace
        self.d_max = 2 * self.gamma
        self.theta = self._invert_deviation(self.d_max)
        thetas = self.theta * np.ones(self.dim_princ_sub//2)
        self.Rb_theta = linalg.block_diag(
            *[self.r_theta(thetas[i]) for i in range(self.dim_princ_sub//2)]
        )
        return self
        
    def distort(self, Xok):
        # genrate a random rotation matrix of angle theta acting 
        # on the principal subspace
        Q_ = np.random.normal(size=(self.dim_princ_sub, self.dim_princ_sub))    
        Q = linalg.orth(Q_)
        R_theta = Q @ self.Rb_theta @ Q.T
    
        loks_sqrt = np.sqrt(self.loks)
        lokn_sqrt = np.sqrt(self.lokn)
        # rotate sqrt of eighenvalues and build marix C
        lkos_sqrt = R_theta @ loks_sqrt
        lko_sqrt = np.concatenate([lkos_sqrt, lokn_sqrt])
        C = self.Uok @ np.diag(lko_sqrt/np.sqrt(self.lok)) @ self.Uok.T
            
        Xko = Xok @ C
        
        return Xko
    
    
class PrincipalSubspaceAlteration(CovarianceAlterations):

    def fit(self, Xok):
        super().fit(Xok)
        # rotation matrix of angle theta acting on the principal subspace
        # self.gamma = np.sum(self.lok) / self.n # energy per component of the normal signal (=1 for standardized data)
        self.d_max = 2 * self.energy_per_component
        self.theta = self._invert_deviation(self.d_max)
        thetas = self.theta * np.ones(self.n//2)
        self.Rb_theta = linalg.block_diag(
            *[self.r_theta(thetas[i]) for i in range(self.n//2)]
        )
        return self
        
    def distort(self, Xok):
        # genrate a random rotation matrix of angle theta 
        # changing the principal subspace
        Q_ = np.random.normal(size=(self.n, self.n))    
        Q = linalg.orth(Q_)
        R_theta = Q @ self.Rb_theta @ Q.T
        C = R_theta.T
            
        Xko = Xok @ C
        
        return Xko
    
