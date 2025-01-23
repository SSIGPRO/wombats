from typing import Any
import numpy as np

def deviation(Xok: np.ndarray, Xko: np.ndarray) -> float:
    """
    A Monte Carlo Estimation of the deviation between two datasets.
    
    Parameters:
        Xok (np.ndarray): A 2D array of normal data with shape (N, n).
        Xko (np.ndarray): A 2D array of anomalous data with shape (N, n).
    
    Returns:
        float: Mean squared deviation between the datasets.
    """
    n = Xok.shape[-1]
    return np.mean(np.sum((Xok - Xko)**2, axis=1) / n)

class Anomaly():
    '''
    Parent class defining a template for 
    the single anomalie class 
    '''
    def __init__(self, delta: float):
        """
        Initializes the AnomalyBase class with a distortion parameter.
        
        Parameters:
            delta (float): The distortion parameter.
        """
        self.delta = delta
        
    def _invert_deviation(self, delta: float) -> float:
        """
        Example method to invert deviation.
        
        Parameters:
            delta (float): The delta value to invert.
        
        Returns:
            float: The parameter value giving deviation delta.
        """
        return parameter
    
    def fit(self, Xok: np.ndarray) -> "Anomaly":
        """
        Fits the anomaly model to the normal data to be distorted.
        
        Parameters:
            Xok (np.ndarray): A 2D array with shape (N, n), where:
                - N is the number of instances
                - n is the dimensionality of each instance
        
        Returns:
            Anomaly: The fitted anomaly model.
        """
        self.n = Xok.shape[-1]
        return self
        
    def distort(self, Xok: np.ndarray) -> np.ndarray:
        """
        Applies the anomaly to the given dataset.
        
        Parameters:
            X (np.ndarray): A 2D array with shape (N, n) or 1D array with shape (, n) 
        containing the normal data.
        
        Returns:
            np.ndarray: Distorted data.
        """

        return Xko
