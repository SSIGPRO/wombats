import numpy as np
from scipy import signal
from wombats.detectors._base import Detector
from wombats.detectors._auto_regressive import burg


class MD(Detector):
    """
    Mahalanobis Distance (MD) detector. Computes the Mahalanobis distance between test data
      and the fitted data distribution.
    """
    
    def __init__(self):
        """
        Initialize the MD detector.
        """
        super().__init__(None)
        
    def fit(self, X_train: np.ndarray):
        """
        Fit the MD detector to the training data.
        
        :param X_train: A 2D array with shape (N, n), containing the training set of normal data. The
        data is expected to be properly scaled: 0 mean and average per sample energy equal 1.
        :return: self (for compatibility with scikit-learn API).
        """
        super().fit(X_train)
        N = X_train.shape[-2]
        self.Sok = 1/(N-1) * X_train.T @ X_train
        eigvals, eigvecs = np.linalg.eigh(self.Sok)
        self.lok = eigvals[::-1] # Eigenvalues in descending order
        self.Uok = eigvecs[:, ::-1] # Corresponding eigenvectors
        return self
        
    def score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute the Mahalanobis distance for the test data.
        
        :param X_test: A 2D array with shape (N, n), containing the test dataset.
        :return: A 1D array with shape (, N) of Mahalanobis distances for each test instance.
        """
        projections = X_test @ self.Uok   
        return np.sqrt(np.sum(projections**2 / self.lok, axis=1))
    
    
class AR(Detector):
    """
    Autoregressive (AR) model adapted for window-based anomaly detection. Fits an AR model to the data and computes prediction errors.

    Attributes:
    - p (int): Order of the AR model.
    - ar_params (np.ndarray): Autoregressive parameters of shape (p,).
    - acf (np.ndarray): Autocorrelation function of shape (p,).
    - pacf (np.ndarray): Partial autocorrelation function of shape (p,).
    - pacf_sigma2 (float): Variance of the partial autocorrelation estimation error.
    """

    def __init__(self, p: int, ar_params: np.ndarray = None):
        """
        Initialize the AR detector.
        
        :param p: Order of the AR model.
        :param ar_params: Optional. Initial autoregressive parameters of shape (p,).
        """
        self.p = p              # model order
        self.ar_params = ar_params  # autoregressive parameters
        self.acf = None             # autocorrelation function
        self.pacf = None            # partial autocorrelation function
        self.pacf_sigma2 = None     # pacf estimation error
        
    def fit(self, x: np.ndarray):
        """
        Fit (estimate) the AR model parameters from the data.
        
        :param x: A 1D or 2D array containing the data to fit the model.
        :return: self (for compatibility with scikit-learn API).
        :raises ValueError: If the specified method is not supported.
        """
        ar_params, acf, pacf, pacf_sigma2 = burg(x, ar_order=self.p)
        self.ar_params = ar_params
        self.acf = acf
        self.pacf = pacf
        self.pacf_sigma2 = pacf_sigma2
    
        return self
        
    def _predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Perform 1-step ahead prediction using the AR model.

        :param x: A 2D array with shape (N, n) containing the test data.
        :return: A 2D array with shape (N, n) of predicted values.
        """
        Y = signal.lfilter(self.ar_params, 1, X_test)
        Y[..., :self.p-1] = np.nan

        return Y

    def score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute the prediction error for the test data.
        
        :param X_test: A 2D array of shape (N, n) containing the test data.
        :return: A 1D array with shape (, N) of prediction errors for each test instance.
        """
        N, n = X_test.shape[0], X_test.shape[1]
        X_pred = self._predict(X_test)
        X_pred = np.c_[np.nan * np.ones((N, 1)), X_pred[:, :-1]]
        return np.sum((X_test[:, self.p:] - X_pred[:, self.p:])**2, axis=1) / (n-self.p)