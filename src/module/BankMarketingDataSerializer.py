class BankMarketingDataSerializer(object):
    def __init__(self,serializer=None):
        self.serializer = serializer
        
    def save(self,estimatorObject=None,filename="model.saved",location=""):
            
        dbfile = open(filename,'ab')
        self.serializer.dump(estimatorObject,dbfile)
        dbfile.close()
        
        return True
    
    def load(self,filename=None):
        
        dbfile = open(filename,'rb')
        modelObject = self.serializer.load(dbfile)
        dbfile.close()
        
        return modelObject