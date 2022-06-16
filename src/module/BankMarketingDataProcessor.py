from math import floor
import pandas as pd

class BankMarketingDataProcessor():
    
    def __init__(self,data=None):
        self.data = data
        
        self.train_data = None
        self.test_data = None
        
        self.clustering_train_data = None
        self.clustering_test_data = None
        
    def clean(self):
        return
    
    def transform(self):
        return
    
    def numerize(self,X=None,categorical_columns=None):
        return pd.get_dummies(X, columns=categorical_columns)
    
    def engineer(self):
        
        categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day", "poutcome"]
        
        bankData_X = self.data.loc[:,self.data.columns != 'y']
        
        numerical_data = self.numerize(X=bankData_X,categorical_columns=categorical_columns)
        
        RTU_Data = pd.DataFrame(numerical_data)
        RTU_Data['y'] = self.data.y
        
        RTU_Data.to_csv("./data/pre_processed_bank_data.csv")
        
        return RTU_Data
    
    def split(self):
        
        data = self.getRTUData()
        
        y_yes = data[data.y == 'yes']
        y_no = data[data.y == 'no']
        
        y_yes_length = len(y_yes)
        y_no_length = len(y_no)
        
        train_y_yes = y_yes.iloc[:floor(0.7*y_yes_length)]
        train_y_no = y_no.iloc[:floor(0.7*y_no_length)]
        
        test_y_yes = y_yes.iloc[floor(0.7*y_yes_length):]
        test_y_no = y_no.iloc[floor(0.7*y_no_length):]
        
        self.train_data = pd.concat([train_y_yes,train_y_no])
        self.test_data = pd.concat([test_y_yes,test_y_no])
        
        return
    
    def getRTUData(self):
        return self.engineer()
    
    def getTrainData(self):
        
        self.split()
        
        X_train = self.train_data.loc[:,self.train_data.columns != 'y']
        y_train = self.train_data['y']
        
        return {"X_train":X_train,"y_train":y_train}
    
    def getTestData(self):
        
        self.split()
        
        X_test = self.test_data.loc[:,self.test_data.columns != 'y']
        y_test = self.test_data['y']
        
        return {"X_test":X_test,"y_test":y_test}
    
    def getClusteringData(self):
        
        clustering_data = self.getRTUData()

        clustering_data = clustering_data.loc[:,clustering_data.columns != 'y']
        
        clustering_data.to_csv("./data/clustering_bank_data.csv")
                
        return clustering_data
        
        