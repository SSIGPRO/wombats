import numpy as np 

class Dataset():
    
    def __init__(self, Xok=None, SNRdB=None):
        
        self.values = Xok
        # shape of the dataset
        self.N, self.n = Xok.shape[0], Xok.shape[1]
        
        # signal to noise ration fo the dataset
        self.SNRdB = SNRdB
        
        # statistics of the dataset
        self.std, self.mean = Xok.std(), Xok.mean()
        self.Sok = 1/(self.N-1) * Xok.T @ Xok
        eigvals, eigvecs = np.linalg.eigh(self.Sok)
        self.lok = eigvals[::-1]
        self.Uok = eigvecs[:, ::-1]
        self.energy_per_component = np.sum(self.lok) / self.n
        
        # infer the dimension of the principal subspace (PS) k
        singvalues = np.sqrt(self.lok)
        if self.SNRdB is None:
            threshold = 2.858 * np.median(singvalues)
        else:
            sigma = 10**(-self.SNRdB/20)
            threshold = 4/np.sqrt(3) * sigma * np.sqrt(self.n)
        
        dim_princ_sub = np.sum(singvalues > threshold)
        if dim_princ_sub%2:
            dim_princ_sub = dim_princ_sub+1
        self.dim_princ_sub = dim_princ_sub
        
        # principal subspace (PS) components
        self.Uoks = self.Uok[:, :self.dim_princ_sub]
        self.loks = self.lok[:self.dim_princ_sub]
        # noise subspace (NS) components
        self.Uokn = self.Uok[:, self.dim_princ_sub:]
        self.lokn = self.lok[self.dim_princ_sub:]
        # fraction of the signal energy contained in the PS
        self.gamma = np.sum(self.loks) / self.n