import numpy as np
from detectors._base import Detector

class Feature(Detector):
    def __init__(self):
        super().__init__(None)
    def fit(self, X_test):
        return self
    def score(self, X_test):
        return feature
    
class pk_pk(Feature):
    def score(self, X_test):
        return np.max(X_test, axis=-1) - np.min(X_test, axis=-1) 
    
    
class TV(Feature):
    def score(self, X_test):
        return np.sum(np.abs(np.diff(X_test, axis=-1)), axis=-1)
    
    
class ZC(Feature):
    def _zero_crossing(self, x):
        zerocross = np.diff(np.sign(x[x!=0]))
        zerocross = np.floor(0.5*np.abs(zerocross))
        zerocross = np.sum(zerocross).astype(int)
        return zerocross
    def score(self, X_test):
        return np.array([self._zero_crossing(x) for x in X_test])

    
class energy(Feature):
    def score(self, X_test):
        return np.sum(X_test**2, axis=1)