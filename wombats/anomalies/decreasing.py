import numpy as np
from wombats.anomalies._base import Anomaly
from scipy import optimize

class Decreasing(Anomaly):
    def __init__(self, delta, fs=1):
        super().__init__(delta)
        self.p = None
        
    def _deviation(self, p, Xok):
        # compute deviation through Monte Carlo
        deviation = np.mean(
            np.sum(
                (Xok - self._distort(Xok, p))**2, axis=1
            )
        )/self.n
        return deviation
        
    def _invert_deviation(self, Xok):
        # limit cases
        if self.delta >= 1:
            p = 1         
        elif self.delta == 0:
            p = 0
         # invert deviation 
        else: 
            delta_p = lambda p : self._deviation(p, Xok) - self.delta
            # p = optimize.brentq(delta_p, a=0.01, b=0.99)
            p = optimize.brentq(delta_p, a=2/self.n, b=(self.n-2)/self.n)
        return p
        
    def fit(self, Xok):
        super().fit(Xok)
        # p corresponding to deviation delta
        self.p = self._invert_deviation(Xok)
        return self
        
    def _distort(self, Xok, p):
        self.p = p
        distorted = self.distort(Xok)
        return distorted
    
    def distort(self, Xok):
        
        return distorted
    

class Clipping(Decreasing): 
    def distort(self, Xok):
        # limit cases
        if self.p == 1:
            return np.zeros(Xok.shape)      
        elif self.p == 0:
            return Xok     
        else:
            ndim = Xok.ndim
            if Xok.ndim == 1:
                Xok = Xok.reshape(1, self.n)
            Xko = np.nan * np.ones(Xok.shape)
            i = int(self.p * self.n) + 1
            # threshold for each instance
            th = np.sort(np.abs(Xok), axis=-1)[..., i, None]
            idx = np.abs(Xok) > th
            Xko = np.sign(Xok) * idx * th
            Xko[~idx] = Xok[~idx]
        if ndim==1:
            Xko = Xko.reshape(self.n)
        return Xko
    
    
class DeadZone(Decreasing):
    def distort(self, Xok):
        # limit cases
        if self.p == 1:
            return np.zeros(Xok.shape)      
        elif self.p == 0:
            return Xok     
        else:
            ndim = Xok.ndim
            if ndim == 1:
                Xok = Xok.reshape(1, self.n)
            Xko = np.nan * np.ones(Xok.shape)
            i = int(self.p * self.n) + 1
        # threshold for each instance
            th = np.sort(np.abs(Xok), axis=-1)[..., i, None]
            idx = np.abs(Xok) > th
            Xko = np.zeros(Xok.shape)
            Xko[idx] = Xok[idx]
        if ndim==1:
            Xko = Xko.reshape(self.n)
        return Xko