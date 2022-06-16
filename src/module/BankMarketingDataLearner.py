from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans
from sklearn.cluster import dbscan

class BankMarketingDataLearner(object):
    
    def __init__(self,processor=None,serializer=None):
        self.processor = processor
        self.serializer = serializer
        
    def createKNeighborsModel(self):
        return
    
    def createSVMModel(self):
        return
    
    def createDecisionTreeModel(self):
        return;
    
    def createRandomForestModel(self):
        return
    
    def createKMeansModel(self):
        return
    
    def createDBScanModel(self):
        return
    
    def predict(self):
        return
    
    def predict_proba(self):
        return