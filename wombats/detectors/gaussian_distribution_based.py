import numpy as np
from detectors._base import Detector


class MD(Detector):
    def __init__(self):
        super().__init__(None)
    def fit(self, X_train):
        super().fit(X_train)
        N = X_train.shape[-2]
        self.Sok = 1/(N-1) * X_train.T @ X_train
        eigvals, eigvecs = np.linalg.eigh(self.Sok)
        self.lok = eigvals[::-1]
        self.Uok = eigvecs[:, ::-1]
        return self
        
    def score(self, X_test):
        # compute the projections on the major subspace
        projections = X_test @ self.Uok   
        return np.sqrt(np.sum(projections**2 / self.lok, axis=1))