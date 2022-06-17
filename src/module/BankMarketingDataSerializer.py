class BankMarketingDataSerializer(object):
    def __init__(self,serializer=None):
        self.serializer = serializer
        
    def save(self,estimatorObject=None,filepath="model.saved"):
            
        dbfile = open(filepath,'ab')
        self.serializer.dump(estimatorObject,dbfile)
        dbfile.close()
        
        return True
    
    def load(self,filepath=None):
        
        dbfile = open(filepath,'rb')
        modelObject = self.serializer.load(dbfile)
        dbfile.close()
        
        return modelObject