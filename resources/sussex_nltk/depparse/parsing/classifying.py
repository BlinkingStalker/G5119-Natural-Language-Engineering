'''
Created on 5 Jun 2012

@author: Andy
'''
import datetime as dt

from ..liblinear import liblinearutil as liblinear

def selector(style):
    if style=='liblinear':
        return LiblinearClassifier()
    
def current_time():
        return dt.datetime.ctime(dt.datetime.now())
    
class Classifier(object):
    '''
    Interface representing a classifier. 
    - All functions must be overidden in a subclass.
    - If classifier requires more information than binary sparse 
      feature vectors, then this will require modification of other
      modules. Specifically:
          - this module must include a selector function for which 
            data type to use for the feature vector
          - A generic interface for the datatype must be constructed
            and the "get_feature_vector" function in parsing_functions
            module should be modified to use the interface.
    '''
    
    def __init__(self,*args):
        pass
    
    def train(self,training_file,model_savefile,options=''):
        '''
        Train a classifier using the feature vectors in training_file,
        passing options to the classifier. Save the classifier model 
        to model_savefile. 
        '''
        pass
    
    def predict(self,feature_vector,options=''):
        '''
        Given the single feature_vector, and the classifier options
        return the predicted label. 
        '''
        pass
    
    def decision_scores_list(self):
        '''
        From the last classification prediction, return the list of 
        pairs of labels and their score.
        '''
        pass
    
    def save(self,savefile):
        '''Save classifier model'''
        pass
    
    def load(self,savefile):
        '''Load classifier model'''
        pass
    
class LiblinearClassifier(Classifier):
    
    def __init__(self,*args):
        super(LiblinearClassifier,self).__init__(args)
        self.model = None
        self.decision_values = None
        
    def train(self,training_file,model_savefile,options=''):
        print "parser: %s> Loading vectors..." % current_time()
        y,x = liblinear.svm_read_problem(training_file)
        print "parser: %s> Training SVM model..." % current_time()
        self.model = liblinear.train(y,x,options)
        print "parser: %s> Saving SVM model..." % current_time()
        self.save(model_savefile)
        return self.model    
        
    def predict(self,feature_vector,options=''):
        label,_,self.decision_values = liblinear.predict([0],[feature_vector.vector_dict()],self.model)
        return int(label[0])
    
    def decision_scores_list(self):
        label_order = self.model.get_labels()
        return ((label_order[i],score) for i,score in enumerate(self.decision_values[0]))
    
    def save(self,savefile):
        liblinear.save_model(savefile,self.model)
    
    def load(self,savefile):
        self.model = liblinear.load_model(savefile)
        
