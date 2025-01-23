import numpy as np
from sklearn.metrics import roc_auc_score

def AUC(scores: np.ndarray) -> float:
    """
    Calculate the AUC metric based on the scores.
    
    :param scores: A 1D with shape (, 2N) array containing
      the N scores for normal data and N scores for anomalous data.
    :return: The complement of the ROC AUC score.
    """
    N = len(scores)//2
    labels = np.concatenate((np.zeros(N), np.ones(N)))
    return 1-roc_auc_score(labels, scores)
    
def P_D(scores: np.ndarray) -> float:
    """
    Calculate the P_D metric based on the AUC.
    
    :param scores:  A 1D with shape (, 2N) array containing
      the N scores for normal data and N scores for anomalous data.
    :return: The probability of detection metric.
    """
    auc = AUC(scores)
    return 0.5 + np.abs(0.5 - auc)
    
class Detector():
    """
    A generic detector class for anomaly detection.

    Attributes:
    - param: Configuration parameter(s) for the detector.
    """
    
    def __init__(self, param: any) -> None:
        """
        Initialize the detector with parameters.
        
        :param param: Configuration parameter(s) for the detector.
        """
        self.param = param
    
    def fit(self, Xok_train: np.ndarray) -> None:
        """
        Fit the detector to the training dataset.
        
        :param Xok_train: A 2D array with shape (N, n) containing the training set of normal data.
        :return: None.
        :raises NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("The 'fit' method must be implemented by subclasses.")
    
    def score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for the test dataset 
        
        :param X_test: A 2D array with shape (2N, n) containing containing N normal Xok instances and N Xko instances.
        :return: A 1D with shape (, 2N) array containing the anomaly scores.
        :raises NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("The 'score' method must be implemented by subclasses.")
        
    def test(self, X_test: np.ndarray, metric: str) -> float:
        """
        Test the detector using a specified evaluation metric.
        
        :param X_test: A 2D array with shape (2N, n) containing containing N normal Xok instances and N Xko instances.
        :param metric: A string specifying the metric ('AUC' or 'P_D').
        :return: The computed metric value.
        :raises ValueError: If the metric is invalid.
        """
        scores = self.score(X_test)
        # test the detector in terms of AUC or P_D
        if metric == 'AUC':
            return AUC(scores)
        elif metric == 'P_D':
            return P_D(scores)
        else:
            raise ValueError(f"Invalid metric '{metric}'. Supported metrics: 'AUC', 'P_D'.")

    