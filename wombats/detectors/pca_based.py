import numpy as np
from wombats.detectors._base import Detector


class PCA(Detector):
    """
    Principal Component Analysis (PCA)-based detector.
    Performs dimensionality reduction and anomaly scoring based on projections onto principal subspaces.
    """
    def __init__(self, k: int):
        """
        Initialize the PCA detector with the number of principal components.

        :param k: Number of principal components to retain.
        """
        # TODO: implement an automatic initialization according to SVD criterion if the provided k is None
        self.k = k
        
    def fit(self, X_train: np.ndarray):
        """
        Fit the PCA model to the training normal data.

        :param X_train: A 2D array of shape (N, n) containing training normal data.
        :return: self (for compatibility with scikit-learn API).
        """
        N = X_train.shape[-2]
        self.Sok = 1/(N-1) * X_train.T @ X_train
        eigvals, eigvecs = np.linalg.eigh(self.Sok)
        self.lok = eigvals[::-1]
        self.Uok = eigvecs[:, ::-1]

        # Define principal componenets (major) subspace
        self.Uok_major = self.Uok[:, :self.k]
        self.lok_major = self.lok[:self.k]
        return self
        
    def score(self, X_test: np.ndarray):
        """
        Compute the projections of the test data onto the major subspace.

        :param X_test: A 2D array of shape (N, n), containing test data.
        :return: None. Sets the projections as an attribute for further computations.
        """
        self.projections = X_test @ self.Uok_major

class SPE(PCA):
    """
    Squared Prediction Error (SPE)-based detector.
    Computes the energy difference between the input data and its projection onto the major subspace.
    """
    def score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute the SPE score for the test data.

        :param X_test: A 2D array with shape (N, n), containing test data.
        :return: A 1D array with shape (, N) of SPE scores for each test instance.
        """
        super().score(X_test)
        # compute SPE score as the energy difference
        self.energy_proj = np.sum(self.projections**2, axis=1)
        self.energy = np.sum(X_test**2, axis=1)
        return self.energy - self.energy_proj
    
class T2(PCA):
    """
    Hotelling's T-squared (T^2)-based detector.
    Computes a squared Mahalanobis distance inside the major subspace.
    """
    def score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute the T^2 score for the test data.

        :param X_test: A 2D array with shape (N, n), containing test data.
        :return: A 1D array with shape (, N) of T^2 scores for each test instance.
        """
        super().score(X_test)
        return np.sum(self.projections**2 / self.lok_major, axis=1)