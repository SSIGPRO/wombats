import numpy as np
from wombats.anomalies._base import Anomaly

class Disturbance(Anomaly):
    """
    Initialize the Disturbance (power-increasing) anomaly.
    
    :param delta: The target deviation value.
    """
    def __init__(self, delta):
        super().__init__(delta)
        self.a = self._invert_deviation()
    
    def _invert_deviation(self) -> float:
        return np.sqrt(self.delta)
    
    def generate(self, N: int):
        """
        Generate disturbance (to be implemented by subclasses).
        
        :param N: The number of instances to generate.
        :return: The generated disturbance signal.
        """
        return disurbance
    
    def distort(self, Xok: np.ndarray) -> np.ndarray:
        """
        Apply disturbance to the signal.
        
        :param Xok: A 2D array with shape (N, n) or 1D array with shape (, n) 
        containing the normal data.
        :return: The distorted signal.
        """
        N = 1 if Xok.ndim == 1 else Xok.shape[-2]
        disturbance = self.generate(N)
        return disturbance + Xok 
    

class Constant(Disturbance):
    
    def generate(self, N: int) -> np.ndarray:
        """
        Generate constant disturbance.
        
        :param N: The number of instances to generate.
        :return: The generated constant disturbance.
        """
        # Sign of the constant distrubance
        sign_array = np.random.choice(np.arange(-1, 2, 2), size=N)[..., None]
        disturbance = self.a * sign_array * np.ones(shape=(N, self.n))
        if N == 1: disturbance = disturbance[0]
        return disturbance
    
    
class Step(Disturbance):
    
    def generate(self, N: int) -> np.ndarray:
        """
        Generate step disturbance.
        
        :param N: The number of instances to generate.
        :return: The generated step disturbance.
        """
        # Randomly select rising or falling step
        r_array = np.random.choice(np.arange(-1, 2, 2), size=N)[..., None]
        # Step position in the middle of the instance
        j_array = (self.n//2) * np.ones(shape=N, dtype=int)
        # Sign of the constant distrubance
        sign_array = np.random.choice(np.arange(-1, 2, 2), size=N)[..., None]
        
        disturbance =  np.array(
            [np.concatenate([
                np.sqrt(self.n/j) * 0.5*(1 + r) * np.ones(shape=j),
                np.sqrt(self.n/(self.n-j)) * 0.5*(1 - r) * np.ones(shape = self.n-j)
        ])
             for j, r in zip(j_array, r_array)])
        disturbance = self.a * sign_array * disturbance
        
        if N == 1: disturbance = disturbance[0]
        return disturbance
    
    
class Impulse(Disturbance):

    def generate(self, N: int) -> np.ndarray:
        """
        Generate impulse disturbance.
        
        :param N: The number of instances to generate.
        :return: The generated impulse disturbance.
        """
        # Impulse position in the middle of the instance
        j_array = (self.n//2) * np.ones(shape=N, dtype=int)
        # Sign of the constant distrubance
        sign_array = np.random.choice(np.arange(-1, 2, 2), size=N)[..., None]
        
        disturbance = np.zeros(shape=(N, self.n))
        idx = np.arange(N)
        disturbance[(idx, j_array[idx])] = np.sqrt(self.n)
        disturbance = self.a * sign_array * disturbance
        
        if N == 1: disturbance = disturbance[0]
        return disturbance
    

class GWN(Disturbance):
        
    def generate(self, N: int) -> np.ndarray:
        """
        Generate Gaussian white noise (GWN) disturbance.
        
        :param N: The number of instances to generate.
        :return: The generated GWN disturbance.
        """
        disturbance = self.a * np.random.normal(size=(N, self.n))
        if N == 1:
            disturbance = disturbance[0]      
        return disturbance
    
    
class GNN(Disturbance):
        
    def generate(self, N: int) -> np.ndarray:
        """
        Generate Gaussian noise with a narrowband spectrum (GNN).
        
        :param N: The number of instances to generate.
        :return: The generated GNN disturbance.
        """
        fs = 1
        
        f0_array = np.round(np.random.uniform(0, fs/2, size=N), 4)
        band_array = np.round(np.array([np.random.uniform(0, fs/2 - np.abs(2*f0 - fs/2))
                                       for f0 in f0_array]), 4)
        
        # define the lags        
        Cols = np.ones((self.n, self.n)) * np.arange(0, self.n)
        Rows = Cols.T
        Lags = Rows - Cols
            
        disturbance = np.zeros((N, self.n))

        for i in range(N):
            f0, band = f0_array[i], band_array[i]
            check = False
            
            while not check:
                cov=np.cos(2*np.pi*Lags*f0) * np.sinc(Lags*band)
                try: 
                    w, V = np.linalg.eigh(cov)
                    w[w < 0] = 0 # Ensure positive semi-definite covariance matrix
                    check = True
                except:
                    # In case of error generate a new pair of f0 and band
                    f0 = np.random.uniform(0, fs/2)
                    band = np.random.uniform(0, fs/2 - np.abs(2*f0 - fs/2))
                    
            disturbance[i] = np.random.randn(1, self.n) @ (V * np.sqrt(w)).T 
            
        disturbance = self.a * disturbance
        
        if N == 1:
            disturbance = disturbance[0]
                
        return disturbance    
