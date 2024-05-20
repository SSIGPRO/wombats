import numpy as np
from detectors._base import Detector
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class SklearnDetector(Detector):
        
    def fit(self, X_train):
        self.detector = make_pipeline(
                StandardScaler(),
                Detector(params)
        )
        self.detector.fit(X_train)
        return self
    
    def score(self, X_test):
        return self.detector.decision_function(X_test)
    
    
class OCSVM(SklearnDetector):
    def __init__(self, kernel, nu):
        self.kernel = kernel
        self.nu = nu
        
    def fit(self, X_train):
        self.detector = make_pipeline(
                StandardScaler(),
                OneClassSVM(kernel=self.kernel, nu=self.nu)
        )
        self.detector.fit(X_train)
        return self
        
    
class LOF(SklearnDetector):
    def __init__(self, h):
        self.h = h # number of neighbors
    
    def fit(self, X_train):
        self.detector = make_pipeline(
                StandardScaler(),
                LocalOutlierFactor(n_neighbors=self.h, novelty=True)
        )
        self.detector.fit(X_train)
        return self
        
    
class IF(SklearnDetector):
    def __init__(self, l):
        self.l = l # number of estimators
        
    def fit(self, X_train):
        self.detector = make_pipeline(
                StandardScaler(),
                IsolationForest(n_estimators=self.l)
        )
        self.detector.fit(X_train)
        return self