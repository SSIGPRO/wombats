import numpy as np
from wombats.detectors._base import Detector

class Feature(Detector):
    def __init__(self):
        """
        Initialize the Feature detector.
        """
        super().__init__(None)
    def fit(self, Xok_train: np.ndarray):
        """
        Fit the model to the training set.

        :param Xok_train: A 2D array with shape (N_train, n) containing the training set of normal data.
        :return: self (for compatibility with scikit-learn API).
        """
        return self

    def score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute scores based on a specific feature. 
        
        :param X_test: A 2D array with shape (N, n) containing the test data.
        :return: A 1D array with shape (, N) containing the feature scores.
        """
        raise NotImplementedError("The 'score' method must be implemented in subclasses.")

    
class pk_pk(Feature):
    """
    Peak-to-peak feature detector. Computes the difference between the maximum and minimum values.
    """
    def score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute the peak-to-peak score for each instance.
        
        :param X_test: A 2D array with shape (N, n) containing the test dataset.
        :return: A 1D array with shape (, N) containing the peak-to-peak scores for each sample.
        """
        return np.max(X_test, axis=-1) - np.min(X_test, axis=-1) 
    
    
class TV(Feature):
    """
    Total Variation (TV) feature detector. Computes the sum of absolute differences between consecutive points.
    """
    def score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute the total variation score for each instance.
        
        :param X_test: A 2D array with shape (N, n) containing the test dataset.
        :return: A 1D array with shape (, N) containing the total variation scores for each instance.
        """
        return np.sum(np.abs(np.diff(X_test, axis=-1)), axis=-1)
    
    
class ZC(Feature):
    """
    Zero Crossing (ZC) feature detector. Counts the number of zero crossings in each instance.
    """
    def _zero_crossing(self, x: np.ndarray) -> int:
        """
        Count the number of zero crossings in a single instance.
        
        :param x: A 1D array with shape (, n) containing a single instance.
        :return: The number of zero crossings.
        """
        zerocross = np.diff(np.sign(x[x!=0]))
        zerocross = np.floor(0.5*np.abs(zerocross))
        zerocross = np.sum(zerocross).astype(int)
        return zerocross

    def score(self, X_test):
        """
        Compute the zero crossing score for each sample.
        
        :param X_test: A 2D array with shape (N, n) containing the test dataset.
        :return: A 1D array with shape (, N) containing the zero crossing scores for each instance.
        """
        return np.array([self._zero_crossing(x) for x in X_test])

    
class energy(Feature):
    """
    Energy feature detector. Computes the energy of each instance
    """
    def score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute the energy score for each instance.
        
        :param X_test: A 2D array with shape (N, n) containing the test dataset.
        :return: A 1D array with shape (, N) containing the energy scores for each instance.
        """
        return np.sum(X_test**2, axis=1)