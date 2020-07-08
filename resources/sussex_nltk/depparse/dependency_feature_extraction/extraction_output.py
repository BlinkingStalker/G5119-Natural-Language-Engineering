'''
Created on Oct 18, 2012

@author: adr27
'''

class FeatureWriter(object):
    
    def __init__(self,file_path=None,file_pointer=None):
        pass
    
    def add_feature(self,word,feature):
        pass
    
    def add_relation(self,word1,relation,word2):
        self.add_feature(word1, ':'.join(relation,word2))
    
    def output_cached(self,file_pointer=None):
        pass
    
    def clear_cached(self):
        pass