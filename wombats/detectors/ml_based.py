import numpy as np
from wombats.detectors._base import Detector
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class SklearnDetector(Detector):
    """
    Base class for scikit-learn anomaly detectors.
    """
        
    def fit(self, X_train: np.ndarray):
        """
        Fit the detector pipeline to the training normal data.

        :param X_train: A 2D array of shape (N, n), containign training normal data
        :return: self
        """
        # detector's pipele template 
        self.detector = make_pipeline(
                StandardScaler(), # Performs per sample scaling 
                Detector(params)
        )
        self.detector.fit(X_train)
        return self
    
    def score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for the test data.

        :param X_test: A 2D array with shape (N, n), containing test data
        :return: A 1D array of anomaly scores with shape (, N) for each test instance.
        """
        return self.detector.decision_function(X_test)
    
    
class OCSVM(SklearnDetector):
    """
    One-Class Support Vector Machine (OCSVM)-based detector.
    """
    def __init__(self, kernel: str = "rbf", nu: float = 0.5):
        """
        Initialize the OCSVM detector with specified kernel and nu parameter. 
        For more details consult scikit-learn documentation. 

        :param kernel: The kernel type to be used in the SVM ('linear', 'poly', 'rbf', 'sigmoid').
        :param nu: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
        """
        self.kernel = kernel
        self.nu = nu
        
    def fit(self, X_train: np.ndarray):
        """
        Fit the OCSVM detector pipeline to the training normal data.

        :param X_train: A 2D array of shape (N, n), containing training normal data.
        :return: self
        """
        self.detector = make_pipeline(
                StandardScaler(),
                OneClassSVM(kernel=self.kernel, nu=self.nu)
        )
        self.detector.fit(X_train)
        return self
        
    
class LOF(SklearnDetector):
    """
    Local Outlier Factor (LOF)-based detector.
    """
    def __init__(self, h: int = 20):
        """
        Initialize the LOF detector with the specified number of neighbors.
        For more details consult scikit-learn documentation.

        :param h: Number of neighbors to use for LOF computation.
        """
        self.h = h # number of neighbors
    
    def fit(self, X_train: np.ndarray):
        """
        Fit the LOF detector pipeline to the training normal data.

        :param X_train: A 2D array of shape (N, n), containing training normal data.
        :return: self
        """
        self.detector = make_pipeline(
                StandardScaler(),
                LocalOutlierFactor(n_neighbors=self.h, novelty=True)
        )
        self.detector.fit(X_train)
        return self
        
    
class IF(SklearnDetector):
    """
    Isolation Forest (IF)-based detector.
    """
    def __init__(self, l: int = 100):
        """
        Initialize the IF detector with the specified number of estimators.

        :param l: Number of estimators (trees) to use in the isolation forest.
        """
        self.l = l # number of estimators
        
    def fit(self, X_train: np.ndarray):
        """
        Fit the IF detector pipeline to the training normal data.

        :param X_train: A 2D array of shape (N, n), containing training normal data.
        :return: self
        """
        self.detector = make_pipeline(
                StandardScaler(),
                IsolationForest(n_estimators=self.l)
        )
        self.detector.fit(X_train)
        return self