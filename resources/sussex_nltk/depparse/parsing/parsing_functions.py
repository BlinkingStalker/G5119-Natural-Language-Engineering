'''
Created on 1 Jun 2012

@author: Andy
'''
import cPickle
import datetime as dt

from ..data_handling.feature_handling import FeatureManager
from ..data_handling.text_handling import TextReader
from ..data_handling.configuration_handling import Configuration

from ..data_structures.sparse import DictSparseVector

import parsing_definitions as parse_defs
import classifying

def current_time():
    return dt.datetime.ctime(dt.datetime.now())

class DependencyParser(object):
    
    def __init__(self):
        pass

        
##############
#
#    Training functions
#
##############

    def train_with_config_file(self,config_file):
        self.train_with_config(Configuration(option_file=config_file))

    def train_with_config(self, c):
        self.train(c['training'],c['training']+"-converted",c['features'],
                   c['feature index'],c['classifier model'],c['file format'],
                   c['classifier train options'],c['classifier type'],c['parse style'])

    def train(self,training_file,               #file containing sentences marked up with dependencies
                   converted_file,              #file in which to save training data which is converted into feature vectors
                   feature_table,               #file containing definitions of features to be extracted
                   feature_indices_file,        #file in which to save an index of (features, ID) pairs
                   classifier_savefile,         #file in which to save the classifier model when trained
                   format_file=None,            #file containing definitions of the format of the sentences (or None for default)
                   classifier_options='',       #string of options to pass to the classifier when training
                   classifier_type='liblinear', #type of classifier to use
                   parse_style='arc-eager'):    #type of parser to use
        '''Train a parser.'''
        self.style = parse_defs.select_style(parse_style)
        print "parser: %s> Converting training data..." % current_time()
        self._convert_training_data(training_file, converted_file, feature_table, feature_indices_file, format_file)
        classifier = classifying.selector(classifier_type)
        classifier.train(converted_file,classifier_savefile,classifier_options)
        print "parser: %s> Training complete." % current_time()
        return classifier
    
    def _convert_training_data(self,training_file,converted_file,feature_table,feature_indices_file,format_file=None):
        '''Convert a file of dependency trees into a file of feature vectors mapped to correct transitions'''
        self.fm = FeatureManager(feature_table)
        with open(converted_file,'w') as out:
            for sentence in TextReader(format_file).sentences(training_file):
                for transition,feature_vector in self._convert_tree(sentence):
                    out.write("%s %s\n" % (transition,feature_vector.sorted_str()))
        self.save_feature_manager(feature_indices_file)
        
    def _convert_tree(self,sentence):
        '''Convert a single sentence dependency tree into a list of feature vectors'''
        training_data = []
        state = self.style.get_new_parse_state()   #Create new parse state
        state.initialise(sentence)                 #Initialise state with new sentence
        self.style.new_training_sentence(sentence) #Allow parse style to initialise required training data for new sent
        while not state.is_terminal(): 
            v = self._get_feature_vector(state)    #Get feature vector representation of state
            transition_id = self.fm.get_class_id(*self.style.optimum_training_transition(state))
            training_data.append((transition_id,v))#Build list of feature ids mapped to appropriate transition id
        return training_data
        
##############
#
#    Parse unseen data
#
##############

    def parse_file_with_config_file(self,config_file):
        self.parse_file_with_config(Configuration(option_file=config_file))

    def parse_file_with_config(self,c):
        self.parse_file(c['input'],c['output'],c['feature index'],
                               c['classifier model'],c['classifier type'],
                               c['classifier predict options'],c['file format'],
                               c['parse style'])

    def parse_file(self,input_file,                  #file containing sentences to be parsed   
                        output_file,                 #file where parsed sentences will be written
                        feature_indices_file,        #file containing index of (ID,feature) pairs (produced in training)
                        classifier_model,            #either instance of class in "classifying" module, or path to saved classifier model
                        classifier_type='liblinear', #type of classifier
                        classifier_options='',       #string containing options to pass to the classifier
                        format_file=None,            #path to file containing specification of the format of the sentences
                        parse_style='arc-eager'):    #parsing style
        '''Parse sentences in a file.'''
        
        self.style = parse_defs.select_style(parse_style)
        self.load_feature_manager(feature_indices_file)
        tr = TextReader(format_file)
        classifier = classifier_model
        if isinstance(classifier_model,basestring):     #If a path is specified instead of a classifier instance
            c = classifying.selector(classifier_type)   #Get appropriate classifier instance
            c.load(classifier_model)                    #load file
            classifier = c
        with open(output_file,'w') as output:
            state = self.style.get_new_parse_state()    #create  parser state 
            for sentence in tr.sentences(input_file,exceptions=set(['deprel','head'])):
                state.initialise(sentence)                  #initialise parser state with new sentence
                self.fm.init_sentence(sentence)             #initialise feature manager with new sentence
                while not state.is_terminal():              #continue finding next transition until state is terminal
                    v = self._get_feature_vector(state)     #get feature vector representation of state
                    transition,label = self.fm.get_transition_tuple(classifier.predict(v,classifier_options)) #predict best transition
                    if not self.style.transition(state,transition,label):                                     #try to perform best transition
                        self._apply_best_transition(state, classifier.decision_scores_list())                 #if not possible, perform next best
                for token in sentence: 
                    #For each token in sentence, make sure it has a head attached, attach to ROOT if not
                    if token.parent is None: token.set_head_root() 
                output.write(tr.sentence_str(sentence)) 
                output.write('\n\n')
                
    def parse_sentence(self,sentence,
                            classifier_options=''):
        state = self.style.get_new_parse_state() 
        state.initialise(sentence)
        self.fm.init_sentence(sentence)
        while not state.is_terminal():
            v = self._get_feature_vector(state)
            transition,label = self.fm.get_transition_tuple(self.classifier.predict(v,classifier_options))
            if not self.style.transition(state, transition, label):
                self._apply_best_transition(state,self.classifier.decision_scores_list())
        for token in sentence:
            if token.parent is None: token.set_head_root()
        return sentence
    
    def setup_parser_for_sentence_object_parsing(self,feature_indices_file,
                                                      classifier_model,
                                                      classifier_type='liblinear',
                                                      parse_style='arc-eager'):
        self.style = parse_defs.select_style(parse_style)
        self.load_feature_manager(feature_indices_file)
        self.classifier = classifier_model
        if isinstance(classifier_model,basestring):     #If a path is specified instead of a classifier instance
            c = classifying.selector(classifier_type)   #Get appropriate classifier instance
            c.load(classifier_model)                    #load file
            self.classifier = c
    
    def _apply_best_transition(self,state,decision_scores_list,style="confidence"):
        '''
        Given the current parser state, and a list of (transition id, score) pairs
        keep trying to apply the next best transition until one succeeds.
        '''
        for transition,label in self._order_best_transitions(decision_scores_list,style):
            if self.style.transition(state,transition,label): 
                return
        raise ValueError("Couldn't apply a transition; this shouldn't be possible")
                  
    def _order_best_transitions(self,decision_scores_list,style):  
        '''
        Given a list of (transition id, score) pairs return the list ordered by score
        '''    
        if style=="confidence":                  
            scores = {}
            full_transitions = {}
            for transition in self.fm.base_trans:
                scores[transition] = 0
                full_transitions[transition] = None
            for tran_id,score in decision_scores_list:
                base,label = self.fm.get_transition_tuple(tran_id)
                score = abs(score)
                if score > scores[base]:
                    scores[base] = score
                    full_transitions[base] = (base,label)
            return (full_transitions[base] for base in sorted(self.fm.base_trans,key=scores.__getitem__,reverse=True))
        elif style=='shift':
            return [('shift',None)]
    
##############
#
#    General Parsing-related Functions
#
##############
    
    def save_feature_manager(self,file):
        '''Save the indexer of IDs'''
        with open(file,'w') as out: cPickle.dump(self.fm,out)
    def load_feature_manager(self,fname):
        with open(fname,'rb') as infile: self.fm = cPickle.load(infile)
    
    def _get_feature_vectorOld(self,state):
        '''
        Return a sparse vector representation of the current parser state.
        '''
        v = DictSparseVector()
        for feature in self.fm.features:
            tokens = self._resolve_addresses(feature.addresses, state)
            if tokens:
                for f_type,value in feature.eval_features(tokens):
                    if value is not None:
                        v.binary_set(self.fm.get_id(feature.addresses,f_type,value)) 
        return v
    
    def _get_feature_vector(self,state,normalise=False):
        v = DictSparseVector()
        for feature in self.fm.features:
            tokens = self._resolve_addresses(feature.addresses, state)
            for f_type,value in feature.eval_features(tokens):
                if value is not None:
                    v.binary_set(self.fm.get_id(feature.addresses,f_type,value)) 
        if normalise: v.binary_normalise()
        return v
    
    def _resolve_addresses(self,addresses,state):
        '''
        Given the state of the parser and the tokens in it, and a string representation
        of the locations of tokens within that state, return the actual Token instances.
        '''
        if len(addresses) == 1:
            return self._resolve_address(addresses[0],state)
        else:
            return [self._resolve_address(ad,state) for ad in addresses]
    
    def _resolve_address(self,address,state):
        index = int(address[-1])
        loc = address[-2]
        try:
            element = state.stack[-(index+1)] if loc == 'stk' else state.buffer[index]
        except IndexError: return None
        i = 3
        while element and i <= len(address):
            next = address[-i];i+=1
            if next.endswith('dep'):
                element = element.get_child(next[0])
            elif next.endswith('sib'):
                element = element.get_sibling(next[0])
            elif next == 'head':
                element = element.parent
            elif next == 'ghead':
                element = None if not element.parent else element.parent.parent
        return element
            
            
    
    