class BankMarketingDataSerializer(object):
    def __init__(self,serializer=None):
        self.serializer = serializer
        
    def save(self,modelObject,filename="model.saved",location=""):
            
        dbfile = open(filename,'ab')
        self.serializer.dump(modelObject,dbfile)
        dbfile.close()
        
        return True
    
    def load(self,filename):
        
        dbfile = open(filename,'rb')
        modelObject = self.serializer.load(dbfile)
        dbfile.close()
        
        return modelObject