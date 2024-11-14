import numpy as np
from wombats.detectors._base import Detector


class PCA(Detector):
    def __init__(self, k):
        # initialize with the number of principal components
        self.k = k
        
    def fit(self, X_train):
        # infer eighenvalues and eighenvectors
        N = X_train.shape[-2]
        self.Sok = 1/(N-1) * X_train.T @ X_train
        eigvals, eigvecs = np.linalg.eigh(self.Sok)
        self.lok = eigvals[::-1]
        self.Uok = eigvecs[:, ::-1]
        # define principal componenets (major) subspace
        self.Uok_major = self.Uok[:, :self.k]
        self.lok_major = self.lok[:self.k]
        return self
        
    def score(self, X_test):
        # compute the projections on the major subspace
        self.projections = X_test @ self.Uok_major

class SPE(PCA):
    def score(self, X_test):
        super().score(X_test)
        # compute SPE score as the energy difference
        self.energy_proj = np.sum(self.projections**2, axis=1)
        self.energy = np.sum(X_test**2, axis=1)
        return self.energy - self.energy_proj
    
class T2(PCA):
    def score(self, X_test):
        super().score(X_test)
        # compute SPE score as a squared Mahalanobis distance
        # inside the major subspace
        return np.sum(self.projections**2 / self.lok_major, axis=1)