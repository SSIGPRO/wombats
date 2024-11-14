import numpy as np
from scipy import signal
from detectors._base import Detector
from detectors._auto_regressive import burg


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
    
    
class AR(Detector):
    """
    Autoregressive (AR) model

    Attributes
    ----------
    p: int,
        order of the AR model
    ar_params: (p,) numpy.ndarray,
        autoregressive parameters
    acf: (p,) numpy.ndarray,
        autocorrelation function
    pacf: (p,) numpy.ndarray,
        partial autocorrelation function
    sigma2: float,
        error variance

    Methods
    -------
    fit:
    predict:
    score:
    """

    def __init__(self, p, ar_params=None):
        self.p = p              # model order
        self.ar_params = ar_params  # autoregressive parameters
        self.acf = None             # autocorrelation function
        self.pacf = None            # partial autocorrelation function
        self.pacf_sigma2 = None     # pacf estimation error
        
    def fit(self, x, method='burg'):
        """
        Fit (estimate) parameters of the model from data.

        Parameters
        ----------
        x: numpy.ndarray,
            dataset
        method: str {'burg'}, optional (default 'burg')
            specify the method adopted to fit parameters to data.

        Returns
        -------
        AR
            autoregressive model
        """
        if method == 'burg':
            ar_params, acf, pacf, pacf_sigma2 = burg(x, ar_order=self.p)
            self.ar_params = ar_params
            self.acf = acf
            self.pacf = pacf
            self.pacf_sigma2 = pacf_sigma2
        
        # elif method == 'yule-walker':
        #     pass
        
        else:
            raise ValueError(f'method {method} not supported')
    
        return self
        
    def predict(self, x):
        """
        1-step prediction

        Parameters
        ----------
        x: 1D or 2D numpy.ndarray,
            dataset

        Returns
        -------
        x.shape numpy.array,
            predicted values
        """
        # TODO: add n-step prediction functionality
        y = signal.lfilter(self.ar_params, 1, x)
        y[..., :self.p-1] = np.nan

        return y

    def score(self, X_test):
        N, n = X_test.shape[0], X_test.shape[1]
        X_pred = self.predict(X_test)
        X_pred = np.c_[np.nan * np.ones((N, 1)), X_pred[:, :-1]]
        return np.sum((X_test[:, self.p:] - X_pred[:, self.p:])**2, axis=1) / (n-self.p)