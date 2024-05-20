import numpy as np
from sklearn.metrics import roc_auc_score


class Detector():
    
    def __init__(self, param):
        pass
    
    def train(self, dataset_train):
        pass
    
    def predict(self, X_test):
        return score
        
    def test(self, scores, metric):
        # test the detector in terms of AUC or P_D
        N = len(scores)//2
        labels = np.concatenate((np.zeros(N), np.ones(N)))
        AUC = 1-roc_auc_score(labels, scores)
        if metric == 'AUC':
            return AUC
        elif metric == 'P_D':
            P_D = 0.5 + np.abs(0.5 - AUC)
            return P_D
        
    