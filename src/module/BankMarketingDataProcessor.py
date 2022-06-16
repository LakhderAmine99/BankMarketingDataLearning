import pandas as pd

class BankMarketingDataProcessor():
    
    def __init__(self,data=None):
        self.data = data
        
        self.train_data = None
        self.test_data = None
        
    def clean(self):
        return
    
    def engineer(self):
        
        categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day", "poutcome"]
        
        bankData_X = self.data.loc[:,self.data.columns != 'y']
        
        
        
        return
    
    def transform(self):
        return
    
    def numerize(self,X=None,categorical_columns=None):
        return pd.get_dummies(X, columns=categorical_columns)
    
    def split(self):
        return
    
    def getRTUData(self):
        return
    
    def getTrainingData(self):
        return
    
    def getTestData(self):
        return
    
    def getClusteringTrainingData(self):
        return
    
    def getClusteringTestData(self):
        return
        
        