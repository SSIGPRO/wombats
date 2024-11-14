import numpy as np
from anomalies._base import Anomaly
from scipy import signal

class Disturbance(Anomaly):
    def __init__(self, delta):
        super().__init__(delta)
        self.a = self._invert_deviation()
    
    def _invert_deviation(self):
        return np.sqrt(self.delta)
    
    def generate(self, N):
        return disurbance
    
    def distort(self, Xok):
        if Xok.ndim == 1:
            N = 1
        else:
            N = Xok.shape[-2]
        disturbance = self.generate(N)
        return disturbance + Xok 
    

class Constant(Disturbance):
    
    def generate(self, N):
        # sign of the constant distrubance
        sign_array = np.random.choice(np.arange(-1, 2, 2), size=N)[..., None]
        disturbance = self.a * sign_array * np.ones(shape=(N, self.n))
        if N == 1:
            disturbance = disturbance[0]
        return disturbance
    
    
class Step(Disturbance):
    
    def generate(self, N):
        # randomly select rising or falling step
        r_array = np.random.choice(np.arange(-1, 2, 2), size=N)[..., None]
        # step position in the middle of the instance
        j_array = (self.n//2) * np.ones(shape=N, dtype=int)
        # sing of the constant distrubance
        sign_array = np.random.choice(np.arange(-1, 2, 2), size=N)[..., None]
        
        disturbance =  np.array(
            [np.concatenate([
                np.sqrt(self.n/j) * 0.5*(1 + r) * np.ones(shape=j),
                np.sqrt(self.n/(self.n-j)) * 0.5*(1 - r) * np.ones(shape = self.n-j)
        ])
             for j, r in zip(j_array, r_array)])
        disturbance = self.a * sign_array * disturbance
        
        if N == 1:
            disturbance = disturbance[0]
        return disturbance
    
    
class Impulse(Disturbance):

    def generate(self, N):
        # impulse position in the middle of the instance
        j_array = (self.n//2) * np.ones(shape=N, dtype=int)
        # sign of the constant distrubance
        sign_array = np.random.choice(np.arange(-1, 2, 2), size=N)[..., None]
        
        disturbance = np.zeros(shape=(N, self.n))
        idx = np.arange(N)
        disturbance[(idx, j_array[idx])] = np.sqrt(self.n)
        disturbance = self.a * sign_array * disturbance
        
        if N == 1:
            disturbance = disturbance[0]
        return disturbance
    

class GWN(Disturbance):
        
    def generate(self, N):
        disturbance = self.a * np.random.normal(size=(N, self.n))
        if N == 1:
            disturbance = disturbance[0]      
        return disturbance
    
    
class GNN(Disturbance):
        
    def generate(self, N):
                
        window='boxcar'
        pass_zero='bandpass'
        num_taps=1001
        length = self.n + num_taps
        fs = 1
        
        f0_array = np.round(np.random.uniform(0, fs/2, size=N), 4)
        band_array = np.round(np.array([np.random.uniform(0, fs/2 - np.abs(2*f0 - fs/2))
                                       for f0 in f0_array]), 4)
        
        # define the lags        
        Cols = np.ones((self.n, self.n)) * np.arange(0, self.n)
        Rows = Cols.T
        Lags = Rows - Cols
            
        disturbance = np.zeros((N, self.n))
        i = 0
        for f0, band in zip(f0_array, band_array):
            try:
                disturbance[i] = np.random.multivariate_normal(
                    mean=np.zeros(self.n),
                    cov=np.cos(2*np.pi*Lags*f0) * np.sinc(Lags*band),
                    size=1)
            except:
                awgn = np.random.normal(size=length)
                taps = signal.firwin(
                    numtaps=num_taps,
                    cutoff=[f0 - band/2, f0 + band/2],
                    width=None,
                    window=window,
                    pass_zero=pass_zero,
                    scale=True,
                    fs=1
                )
                filtered = signal.lfilter(taps, 1.0, awgn)[num_taps : num_taps + self.n] 
                disturbance[i] = filtered/ np.sqrt(np.sum(taps**2))
                
            i = i+1
            
        disturbance = self.a * disturbance
        
        if N == 1:
            disturbance = disturbance[0]
                
        return disturbance    
    

# class GNN(Disturbance):
        
#     def generate(self, N):
                
#         window='boxcar'
#         pass_zero='bandpass'
#         num_taps=1001
#         length = self.n + num_taps
#         fs = 1
#         band = [0, fs/2]
        
#         i = 0
        
#         # generate bandwidths for each distrubance instance
#         fmins_array = np.random.uniform(band[0], band[1], N)
#         fmax_array = np.array(
#             [np.random.uniform(fmin, band[1]) for fmin in fmins_array]
#         )
#         band_array = fmax_array - fmins_array
#         f0_array = (fmins_array + fmins_array)/2
        
#         # define the lags        
#         Cols = np.ones((self.n, self.n)) * np.arange(0, self.n)
#         Rows = Cols.T
#         Lags = Rows - Cols
            
#         disturbance = np.zeros((N, self.n))
#         for f0, band in zip(f0_array, band_array):
#             try:
#                 disturbance[i] = np.random.multivariate_normal(
#                     mean=np.zeros(self.n),
#                     cov=np.cos(2*np.pi*Lags*f0) * np.sinc(Lags*band),
#                     size=1)
#             except:
#                 awgn = np.random.normal(size=length)
#                 taps = signal.firwin(
#                     numtaps=num_taps,
#                     cutoff=[f0 - band/2, f0 + band/2],
#                     width=None,
#                     window=window,
#                     pass_zero=pass_zero,
#                     scale=True,
#                     fs=1
#                 )
#                 filtered = signal.lfilter(taps, 1.0, awgn)[num_taps : num_taps + self.n] 
#                 disturbance[i] = filtered/ np.sqrt(np.sum(taps**2))
                
#             i = i+1
            
#         disturbance = self.a * disturbance
        
#         if N == 1:
#             disturbance = disturbance[0]
                
#         return disturbance