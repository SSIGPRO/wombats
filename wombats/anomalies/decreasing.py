import numpy as np
from wombats.anomalies._base import Anomaly
from scipy import optimize

class Decreasing(Anomaly):
    def __init__(self, delta: float):
        """
        Initialize the Decreasing anomaly.
        
        :param delta: The target deviation value.
        """
        super().__init__(delta)
        self.p = None
        
    def _deviation(self,  p: float, Xok: np.ndarray) -> float:
        """
        Compute the deviation using Monte Carlo simulation.
        
        :param p: The distortion parameter.
        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :return: The computed deviation.
        """
        deviation = np.mean(
            np.sum(
                (Xok - self._distort(Xok, p))**2, axis=1
            )
        )/self.n
        return deviation
        
    def _invert_deviation(self, Xok: np.ndarray) -> float:
        """
        Invert the deviation to solve for the parameter p controlling 
        the probability of distortion of instance samples.
        
        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :return: The parameter p.
        """
        # Handle the limit cases
        if self.delta >= 1:
            p = 1         
        elif self.delta == 0:
            p = 0

        # Invert deviation 
        else: 
            delta_p = lambda p : self._deviation(p, Xok) - self.delta
            p = optimize.brentq(delta_p, a=2/self.n, b=(self.n-2)/self.n)
        return p
        
    def fit(self, Xok)-> "Decreasing":
        """
        Fit the Decreasing anomaly to the normal data.
        
        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :return: The fitted Decreasing object.
        """
        super().fit(Xok)
         # Find p that corresponds to the target deviation (delta)
        self.p = self._invert_deviation(Xok)
        return self
        
    def _distort(self, Xok: np.ndarray, p: float) -> np.ndarray:
        """
        Apply distortion to the normal data based on parameter p.
        
        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :param p: The anomaly parameter controlling the probability of distortion
          of instance samples.
        :return: The distorted signal data.
        """
        self.p = p
        distorted = self.distort(Xok)
        return distorted
    
    def distort(self, Xok: np.ndarray) -> np.ndarray:
        """
        Distort the signal data (to be implemented by subclasses).
        
        :param Xok: TA 2D array with shape (N, n) containing the normal data.
        :return: The distorted (anomalous) data.
        """
        
        return Xko
    

class Clipping(Decreasing): 
    def distort(self, Xok: np.ndarray) -> np.ndarray:
        """
        Apply clipping distortion to the normal data.
        
        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :return: The clipped data.
        """
        # Handle the limit cases
        if self.p == 1:
            return np.zeros(Xok.shape)      
        elif self.p == 0:
            return Xok     
        else:
            ndim = Xok.ndim
            if Xok.ndim == 1:
                Xok = Xok.reshape(1, self.n)

            # Initialize distorted signal
            Xko = np.nan * np.ones(Xok.shape)

            # Determine clipping threshold for each instance
            i = int(self.p * self.n) + 1
            th = np.sort(np.abs(Xok), axis=-1)[..., i, None]

             # Apply clipping
            idx = np.abs(Xok) > th
            Xko = np.sign(Xok) * idx * th
            Xko[~idx] = Xok[~idx]

            if ndim==1:
                Xko = Xko.reshape(self.n)
        return Xko
    
    
class DeadZone(Decreasing):
    def distort(self, Xok: np.ndarray) -> np.ndarray:
        """
        Apply dead-zone distortion to the normal data.
        
        :param Xok: A 2D array with shape (N, n) containing the normal data.
        :return: The distorted data with dead-zone effect.
        """
        # Handle the limit cases
        if self.p == 1:
            return np.zeros(Xok.shape)      
        elif self.p == 0:
            return Xok     
        else:
            ndim = Xok.ndim
            if ndim == 1:
                Xok = Xok.reshape(1, self.n)

            # Initialize distorted signal
            Xko = np.nan * np.ones(Xok.shape)

            # Determine dead-zone threshold
            i = int(self.p * self.n) + 1
            th = np.sort(np.abs(Xok), axis=-1)[..., i, None]

            # Apply dead-zone
            idx = np.abs(Xok) > th
            Xko = np.zeros(Xok.shape)
            Xko[idx] = Xok[idx]
            if ndim==1:
                Xko = Xko.reshape(self.n)
        return Xko