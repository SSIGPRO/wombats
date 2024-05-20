import numpy as np
from sklearn.metrics import roc_auc_score

def AUC(scores):
    N = len(scores)//2
    labels = np.concatenate((np.zeros(N), np.ones(N)))
    return 1-roc_auc_score(labels, scores)
    
def P_D(scores):
        auc = AUC(scores)
        return 0.5 + np.abs(0.5 - auc)
    
class Detector():
    
    def __init__(self, param):
        pass
    
    def fit(self, dataset_train):
        pass
    
    def score(self, X_test):
        return scores
        
    def test(self, X_test, metric):
        scores = self.score(X_test)
        # test the detector in terms of AUC or P_D
        if metric == 'AUC':
            return AUC(scores)
        elif metric == 'P_D':
            return P_D(scores)

    