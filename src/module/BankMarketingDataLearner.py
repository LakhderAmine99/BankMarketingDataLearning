import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
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
        self.y_test = self.processor.getTestData()["y_test"]
        
    def createKNeighborsModel(self):
        
        classifier = KNeighborsClassifier(n_neighbors=5,
                                          weights='distance',
                                          algorithm='auto',
                                          leaf_size=30,
                                          p=2,
                                          metric='minkowski',
                                          metric_params=None,
                                          n_jobs=1)
        
        model = classifier.fit(X=self.X_train,y=self.y_train)
        
        self.serializer.save(estimatorObject=model,filepath="./model/classifier/KNeighborsModel.saved")
        
        return model
    
    def createSVMModel(self):
        
        classifier = SVC(C=1.0,kernel='rbf',gamma='auto',probability=True,random_state=0)
        
        model = classifier.fit(X=self.X_train,y=self.y_train)
        
        self.serializer.save(estimatorObject=model,filepath="./model/classifier/SVMModel.saved")
        
        return model
    
    def createDecisionTreeModel(self):

        classifier = DecisionTreeClassifier(criterion='gini',
                                            splitter='best', 
                                            max_depth=None,
                                            min_samples_split=2, 
                                            min_samples_leaf=1)
        
        model = classifier.fit(X=self.X_train,y=self.y_train)
        
        self.serializer.save(estimatorObject=model,filepath="./model/classifier/DecisionTreeModel.saved")
        
        return model
    
    def createRandomForestModel(self):
       
        classifier = RandomForestClassifier(n_estimators=100,
                                            criterion='gini',
                                            max_depth=5,
                                            min_samples_split=2, 
                                            min_samples_leaf=1,
                                            max_features='auto')
    
        model = classifier.fit(X=self.X_train,y=self.y_train)
        
        self.serializer.save(estimatorObject=model,filepath="./model/classifier/RandomForestModel.saved")
        
        return model
    
    def createKMeansModel(self):
        
        cluster = KMeans(n_clusters=2)
        
        model = cluster.fit(X=self.clustering_data)
        
        self.serializer.save(estimatorObject=model,filepath="./model/cluster/KmeansModel.saved")
        
        return model
    
    def evaluate(self,estimator=None):
                
        _accuracy = accuracy_score(y_true=self.y_test,y_pred=self.predict(estimator=estimator,X=self.X_test))
        _f1_score = metrics.f1_score(y_true=self.y_test,y_pred=self.predict(estimator=estimator,X=self.X_test),pos_label='yes')
        _precision = metrics.precision_score(y_true=self.y_test,y_pred=self.predict(estimator=estimator,X=self.X_test),pos_label='yes')
        _recall = metrics.recall_score(y_true=self.y_test,y_pred=self.predict(estimator=estimator,X=self.X_test),pos_label='yes')
        
        
        return _accuracy,_precision,_f1_score,_recall
    
    def predict(self,estimator=None,X=None):
        return estimator.predict(X)
            
    def predict_proba(self,estimator=None,X=None):
        return estimator.predict_proba(X)