'''
Created on 4 May 2012

@author: Andrew D. Robertson
'''
import os

class Configuration(object):
    '''
    Represents the configuration of a parsing session.
    Contains the inputs and outputs, and parser/classifier setup.
    
    'configdir'        = path to where a configuration directory is/should be to contain config files
    
    'training'         = path to training file
    'input'            = path to file to be parsed
    'output'           = path to file in which to save parsed text
    'features'         = path to file containing feature definitions (feature table)
    'config'           = path to file where this configuration will be saved
    'feature_index'    = path to file for keeping track of feature unique IDs 
    'classifier_model' = path to file in which classifier model will be saved/loaded
    
    'file_format'      = path to file specifying format of sentences, or None for default
    'parse_style'      = style of parser to be used
    'classifier_type'  = type of classifier
    'classifier_train_options'   = options passed to classifier during training
    'classifier_predict_options' = options passed to classifier at prediction time
    
    '''
            
    def __init__(self,option_file=None,examples_path='',**kwargs):
        self.opts = {}
        file_options = {}
        
        if option_file:
            file_options = self._read_in_optionfile(option_file)
            try:
                self.opts['configdir'] = file_options['configdir']
            except KeyError: pass
            
        if kwargs:
            try:
                self.opts['configdir'] = kwargs['configdir']
            except KeyError: pass    
            
        
        configdir =  self.opts['configdir'] if 'configdir' in self.opts else os.path.join(os.getcwd(),'config')
        examples = examples_path if examples_path else os.path.join(os.getcwd(),'examples')
        
        self.opts['training'] = os.path.join(configdir,'training.txt')      #path to training file
        self.opts['input']    = os.path.join(configdir,'input.txt')         #path to file to be parsed
        self.opts['output']   = os.path.join(configdir,'output.txt')        #path to file in which to save parsed text
        self.opts['features'] = os.path.join(examples, 'feature_table.txt') #path to file containing feature definitions
        self.opts['config']   = os.path.join(configdir,'parser_config.txt') #path to file where this configuration will be saved
        self.opts['feature index']    = os.path.join(configdir,'feature_indices_file')#path to file for keeping track of feature unique IDs 
        self.opts['classifier model'] = os.path.join(configdir,'classifier_model')  #path to file in which classifier model will be saved/loaded
        
        self.opts['file format'] = None            #path to file specifying format of sentences, or None for default
        self.opts['classifier type'] = 'liblinear' #type of classifier
        self.opts['classifier train options'] = '-s 4 -c 0.1 -e 0.1 -B -1' #options passed to classifier during training
        self.opts['classifier predict options'] = '' #options passed to classifier at prediction time
        self.opts['parse style'] = 'arc-eager'       #style of parser to be used
        
        for option,value in file_options.iteritems():
            self.opts[option] = value
            
        for option,value in kwargs.iteritems():
            self.opts[option.replace('_',' ')] = value
        
    def __getitem__(self,key):
        return self.opts[key]
    
    def __setitem__(self,key,value):
        self.opts[key] = value
        
    def save_config(self):
        '''
        Save configuration to file
        '''
        config = []
        for option,value in sorted(self.opts.iteritems()):
            config.append("%s=%s" % (option,value))
        with open(self.config_file,'w') as output:
            output.write('\n'.join(config))
            
    def _read_in_optionfile(self,infile):
        options = {}
        with open(infile) as infile:
            for line in infile:
                if line.strip() and not line.startswith('#'):
                    option,value = line.split('=')
                    if value.strip() and not value.startswith('#'):
                        options[option.strip().replace(' ','_')] = value.split('#')[0].strip()
        return options
    
                