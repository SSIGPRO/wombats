import numpy as np
from wombats.anomalies._base import Anomaly
from wombats.anomalies.increasing import Disturbance, GWN, Constant
from scipy import optimize
from scipy import linalg
from scipy import interpolate
from typing import Optional, Union

class Mixing(Disturbance):
    """
    A class that represents a distortion model for mixing data with a disturbance term.
    Inherits from `Disturbance`.
    """ 
    def __init__(self, delta: float):
        """
        Initializes the Mixing class.

        :param delta: The deviation parameter for the disturbance.
        """
        self.delta = delta
    
    def fit(self, Xok: np.ndarray) -> "Mixing":
        """
        Fits the Mixing model to the input data by computing the covariance matrix and energy per component.

        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :return: The fitted Mixing model.
        """
        super().fit(Xok)
        N = Xok.shape[-2]
        self.Sok = 1/(N-1) * Xok.T @ Xok
        self.energy_per_component = np.trace(self.Sok) / self.n # for a standardized normal data is 1
        self.d_max = 2 * self.energy_per_component
        return self
    
    def _invert_deviation(self) -> float:
        """
        Inverts the deviation to solve for the amplitude value controlling 
        the distortion of instance samples.

        :return: The disturbance parameter.
        """
        return np.sqrt( 1 - (1 - self.delta / self.d_max)**2 )
    
    def distort(self, Xok: np.ndarray) -> np.ndarray:
        """
        Distorts the input data by applying a disturbance and scaling based on the energy per component.

        :param Xok: A 2D array with shape (N, n) or 1D array with shape (, n) 
        containing the normal data.
        :return: The distorted data.
        """
        N = 1 if Xok.ndim == 1 else len(Xok)
        self.a = self._invert_deviation()
        disturbance = self.generate(N) # Disturbance with a unit energy per component
        disturbance = np.sqrt(self.energy_per_component) * disturbance # Ensure energy per component equal to that of Xok
        return self.a * disturbance + np.sqrt(1-self.a**2) * Xok 
    
    
class MixingGWN(Mixing):
    def fit(self, Xok: np.ndarray) -> "MixingGWN":
        """
        Fits the MixingGWN model to the input data.

        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :return: The fitted MixingGWN model.
        """
        super().fit(Xok) 
        self.gwn = GWN(1).fit(Xok)
        return self
        
    def generate(self, N: int) -> np.ndarray:
        """
        Generates Gaussian White Noise (GWN) disturbance.

        :param N: The number of samples to generate.
        :return: A 2D array containing the generated GWN disturbance.
        """
        return self.gwn.generate(N)
    
    
class MixingConstant(Mixing):
    def fit(self, Xok: np.ndarray) -> "MixingConstant":
        """
        Fits the MixingConstant model to the normal data.

        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :return: The fitted MixingConstant model.
        """
        super().fit(Xok) 
        self.constant = Constant(1).fit(Xok)
        return self
        
    def generate(self, N: int) -> np.ndarray:
        """
        Generates constant disturbance.

        :param N: The number of samples to generate.
        :return: A 2D array containing the generated constant disturbance.
        """
        return self.constant.generate(N)
    

class TimeWarping(Anomaly):
    
    def __init__(self, delta: float, fs: float = 1):
        """
        Initializes the TimeWarping anomaly model.

        :param delta: The deviation parameter.
        :param fs: The sampling frequency (optional, default is 1).
        """
        super().__init__(delta)
        self.alpha = None
        self.fs = fs
    
    def _bspline3(self, x: float) -> float:
        """
        Computes the 3rd-order B-spline basis function.

        :param x: The input value.
        :return: The computed B-spline value.
        """
        return 1/6 * (np.max([0, 2-np.abs(x)])**3)- (2/3)*np.max([0, 1-np.abs(x)])**3

    def _filt_bspline3(self, k: int) -> float:
        a = -2 + np.sqrt(3)
        return (-6*a)*(a**np.abs(k))/(1-a**2)
                                      
    def _card_spline3(self, x: float, m: int = 10) -> float:
        """
        Computes the card spline for a given value x.

        :param x: The input value for the spline.
        :param m: The range of the spline (optional, default is 10).
        :return: The computed card spline value.
        """
        return np.sum([
            self._filt_bspline3(k) * self._bspline3(x-k)
            for k in np.arange(-m, m)]
        )
    
    def _warpfun(self, tt: np.ndarray, alpha: float) -> np.ndarray:
        """
        Warps the time by applying the warp factor `alpha`.

        :param tt: The time array.
        :param alpha: The warp factor.
        :return: The warped time.
        """
        return (1-alpha)*tt
    
    def _deviation(self, alpha: float) -> float:
        """
        Computes the deviation using sinc interpolation.

        :param alpha: The warp factor.
        :return: The computed deviation.
        """
    
        # Deviation is computed through sinc interpolation
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
    
    def _invert_deviation(self) -> float:
        """
        Inverts the deviation to solve for the warp factor `alpha`.

        :return: The computed warp factor `alpha`.
        """
        # Invert deviation in the range of alpha [0, 1]
        delta_alpha = lambda alpha : self._deviation(alpha) - self.delta
        alpha = optimize.brentq(delta_alpha, a=0, b=1)
        return alpha
        
    def fit(self, Xok: np.ndarray) -> "TimeWarping":
        """
        Fits the TimeWarping model to the input data.

        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :return: The fitted TimeWarping model.
        """
        super().fit(Xok)
        
        N = Xok.shape[-2]
        self.Sok = 1/(N-1) * Xok.T @ Xok
    
        # Find alpha corresponding to deviation level delta
        self.alpha = self._invert_deviation()
        
        # Compute warping matrix
        W = np.zeros((self.n,self.n))
        for j in range(self.n):
            for i in range(self.n):
                W[j, i] = self._card_spline3(
                    self.fs*self._warpfun(j/self.fs, self.alpha) - i
                )
        self.scaling = np.sqrt(np.trace(self.Sok) \
                               / np.trace(W.T @ W @ self.Sok))
        
        return self
        
    def distort(self, Xok: np.ndarray) -> np.ndarray:
        """
        Distorts the input normal data by applying the time warping transformation.

        :param Xok: A 2D array with shape (N, n) or 1D array with shape (, n) 
        containing the normal data.
        :return: The warped and scaled data.
        """
        ndim = 2
        if Xok.ndim == 1:
            ndim = 1
            Xok = Xok.reshape(1, self.n)
        Xko = np.zeros(Xok.shape)
        for i in range(len(Xok)):
            # Interpolate with spline
            spline = interpolate.splrep(np.arange(self.n), Xok[i], s=0, k=3)
            tt = self._warpfun(np.arange(self.n)/self.fs, self.alpha) # Warped time
            Xko[i] = interpolate.splev(tt, spline, der=0)
        # Scale
        Xko = self.scaling * Xko
        if ndim==1:
            Xko = Xko.reshape(self.n)
            
        return Xko
    

class CovarianceAlterations(Anomaly):
    """
    A base class for anomaly detection models that modify the covariance structure of the data.
    """
    def __init__(self, delta: float):
        """
        Initializes the CovarianceAlterations class.

        :param delta: The deviation parameter.
        """
        super().__init__(delta)
        self.theta = None
        self.d_max = None
    
    def _invert_deviation(self, d_max: float) -> float:
        """
        Inverts the deviation to solve for the rotation angle controlling 
        the covariance structure alteration.

        :param d_max: The maximum deviation.
        :return: The rotation angle.
        """
        theta = np.arccos(1-self.delta/d_max)
        return theta
    
    def r_theta(self, theta_: float) -> np.ndarray:
        """
        Computes the standard 2D rotation matrix for a given angle.

        :param theta_: The rotation angle.
        :return: The rotation matrix.
        """
        return np.array([
            [np.cos(theta_), -np.sin(theta_)],
            [np.sin(theta_), np.cos(theta_)]
        ])
        
    def fit(self, Xok: np.ndarray) -> "CovarianceAlterations":
        """
        Fits the CovarianceAlterations model to the normal data by calculating covariance and
         performing eigendecomposition.

        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :return: The fitted CovarianceAlterations model.
        """
        super().fit(Xok)
        # Theta corresponding to deviation delta
        N = Xok.shape[-2]
        self.Sok = 1/(N-1) * Xok.T @ Xok
        eigvals, eigvecs = np.linalg.eigh(self.Sok)
        self.lok = eigvals[::-1]
        self.Uok = eigvecs[:, ::-1]
        
        self.energy_per_component = np.sum(self.lok) / self.n
        return self
    
    
class SpectralAlteration(CovarianceAlterations):
   
    def fit(self, Xok: np.ndarray, SNRdB: Optional[float] = None) -> "SpectralAlteration":
        """
        Fits the SpectralAlteration model, computing the rotation matrix and
          adjusting the energy.

        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :param SNRdB: The Signal-to-Noise Ratio in dB (optional).
        :return: The fitted SpectralAlteration model.
        """
        super().fit(Xok)
        # Infer the dimension of the principal subspace (PS) k
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
        
        # Principal subspace (PS) components
        self.Uoks = self.Uok[:, :self.dim_princ_sub]
        self.loks = self.lok[:self.dim_princ_sub]
        # Noise subspace (NS) components
        self.Uokn = self.Uok[:, self.dim_princ_sub:]
        self.lokn = self.lok[self.dim_princ_sub:]
        # Fraction of the signal energy contained in the PS
        self.gamma = np.sum(self.loks) / self.n
        
        # Rotation matrix of angle theta acting on the principal subspace
        self.d_max = 2 * self.gamma
        self.theta = self._invert_deviation(self.d_max)
        thetas = self.theta * np.ones(self.dim_princ_sub//2)
        self.Rb_theta = linalg.block_diag(
            *[self.r_theta(thetas[i]) for i in range(self.dim_princ_sub//2)]
        )
        return self
        
    def distort(self, Xok: np.ndarray) -> np.ndarray:
        """
        Distorts the normal data by applying a random rotation to the principal subspace.

        :param Xok: A 2D array with shape (N, n) or 1D array with shape (, n) 
        containing the normal data.
        :return: The distorted data with altered eigenvalues.
        """
        # Generate a random rotation matrix of angle theta acting 
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

    def fit(self, Xok: np.ndarray) -> "PrincipalSubspaceAlteration":
        """
        Fits the PrincipalSubspaceAlteration model, computing the rotation matrix for altering the principal subspace.

        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :return: The fitted PrincipalSubspaceAlteration model.
        """
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
        
    def distort(self, Xok: np.ndarray) -> np.ndarray:
        """
        Distorts the normal data by rotating the principal subspace with the computed rotation matrix.

        :param Xok: A 2D array with shape (N, n) or 1D array with shape (, n) 
        containing the normal data.
        :return: The distorted data after altering the principal subspace.
        """
        # Generate a random rotation matrix of angle theta 
        # altering the principal subspace
        # genrate a random rotation matrix of angle theta 
        # changing the principal subspace
        Q_ = np.random.normal(size=(self.n, self.n))    
        Q = linalg.orth(Q_)
        R_theta = Q @ self.Rb_theta @ Q.T
        C = R_theta.T
            
        Xko = Xok @ C
        
        return Xko
    
