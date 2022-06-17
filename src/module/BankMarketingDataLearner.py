from pyexpat import model
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import accuracy_score
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
        
        self.clustering_data = self.processor.getClusteringData()
        
        self.X_train = self.processor.getTrainData()["X_train"]
        self.y_train = self.processor.getTrainData()["y_train"]
        self.X_test = self.processor.getTestData()["X_test"]
        self.X_test = self.processor.getTestData()["y_test"]
        
    def createKNeighborsModel(self):
        return
    
    def createSVMModel(self):
        return
    
    def createDecisionTreeModel(self):
        return;
    
    def createRandomForestModel(self):
        return
    
    def createKMeansModel(self):
        
        cluster = KMeans(n_clusters=2)
        
        model = cluster.fit(X=self.clustering_data)
        
        # self.serializer.save(model,"./src/model/cluster/KmeansModel.saved")
        
        return model
    
    def createDBScanModel(self):
        return
    
    def evaluate(self,estimator=None,X=None,y=None):
        return
    
    def predict(self,estimator=None,X=None):
        return
    
    def predict_proba(self,estimator=None,X=None):
        return