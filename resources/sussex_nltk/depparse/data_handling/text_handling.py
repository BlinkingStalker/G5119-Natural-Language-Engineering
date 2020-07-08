'''
Created on 4 May 2012

@author: Andrew D. Robertson
'''
root_rel = 'root'

class Token(object):

    def __init__(self,atts):
        self.atts = atts 
        assert 'id' in atts,'Tokens must have IDs'
        
        if 'head' not in atts:
            atts['head'] = None
        if 'deprel' not in atts:
            atts['deprel'] = None
        
        self.lchild = None
        self.rchild = None
        self.cchild = None
        
        #Link to the Token object which is head of this token
        self.parent = None 
        
        #Use the "deps" property to get the combination of these two
        self.ldeps = 0
        self.rdeps = 0
        
    def __getitem__(self,key):
        try:
            return self.atts[key]
        except KeyError:
            return None
    
    @property    
    def id(self):
        return self.atts['id']
    @property
    def deps(self):
        return self.ldeps+self.rdeps
    
    def is_root(self):
        return self.atts['id'] == '0'
  
    def __cmp__(self,other):
        return int(self['id']) - int(other['id'])
    
    def dist(self,other):
        '''
        Return the distance between this token and another, in terms of the number of
        words inbetween.
        '''
        return abs(int(self['id']) - int(other['id']))-1
    
    def get_feature(self,type,args=None):
        '''
        Return the value of a feature of this token, features as specified
        by the Format object passed to the TextReader.
        '''
        if type=='deps': return str(self.ldeps+self.rdeps)
        elif type=='ldeps': return str(self.ldeps)
        elif type=='rdeps': return str(self.rdeps)
        elif type=='deprel' or type=='head':
            if self.parent: return self.atts[type]
        elif type in self.atts: return self.atts[type]
        else: return None
        
    def get_child(self,loc):
        if   loc == 'l': return self.lchild
        elif loc == 'r': return self.rchild
        elif loc == 'c': return self.cchild
        
    def get_sibling(self,loc):
        #Note: passing 'c' will return the closest child to the parent, not the closest sibling
        return self.parent.get_child(loc) if self.parent else None
        
    def set_head(self,head,relation):
        self.parent = head
        self.atts['head'] = head['id']
        self.atts['deprel'] = relation
        head._set_dependent(self)
        
    def set_head_root(self,label=None):
        self.atts['head'] = '0'
        self.atts['deprel'] = root_rel if label is None else label
        
    def _set_dependent(self,dependent):
        '''
        Add a new dependent to the token. Update number of dependents,
        leftmost dependent and rightmost dependent.
        '''
        if dependent < self: self.ldeps+=1
        else: self.rdeps+=1
        
        if (not self.lchild) or (dependent<self.lchild): self.lchild = dependent
        if (not self.rchild) or (dependent>self.rchild): self.rchild = dependent
        if (not self.cchild) or (self.dist(dependent)<self.dist(self.cchild)): self.cchild=dependent
        
    def __str__(self):
        #return "%s:%s" %(self.id,self.atts)
        try:
            return "%s:%s" %(self['id'],self['form'])
        except KeyError:
            return "%s:%s" %(self['id'],self.atts)
    
    
class Format(object):
    '''
    NOTE: 'ignore' and 'fill' should not be used as feature names. They are keywords. Read below.
    
    Class representing the format of text files to be read and written.
    Sentences are expected token per line, where each token is a tab-separated (or otherwise)
    list of features. Sentences are separated by a blank line. The format file specifies what
    type and in what order features appear in in input files.
    
    File specifying format should be a text file where:
        1. Each line contains a single term
        2. Each term specifies a feature present in a token line. E.g. "deprel"
        3. The terms are in the order that the corresponding features appear in the token line.
        4. If the term is 'ignore' then this feature will be ignored when reading, and nothing will
           be written in this slot when writing out a token.
        5. If the term is 'fill' then this feature will be ignored when reading, but when writing
           the feature will be listed as an underscore '_'
        6. If "id" is missing as a feature, then it will be added to the beginning, and an id will
           given to each new token in a sentence. This ID will later be outputted when writing the
           tokens.
           
    The default format expects tokens in this style:
        
        ID    FORM     POS    HEAD    DEPREL
    '''
    
    def __init__(self,separator,formatfile=None):
        self.id_present = False
        self.format = self._parse_formatfile(formatfile) if formatfile else self._default_format()
        self.separator = separator
        
    def prepend_id(self):
        self.format.insert(0,'id')
    
    def _default_format(self):
        return ['id','form','pos','head','deprel']
    
    def _parse_formatfile(self,infile):
        format = []
        head_present = deprel_present = False
        with open(infile) as formatfile:
            for line in formatfile:
                line = line.strip().lower()
                if line and not line.startswith('#'):
                    format.append(line)
                    if line=='id': self.id_present    = True
                    elif line=='head': head_present   = True
                    elif line=='deprel':deprel_present= True
        if not head_present: format.append('head')
        if not deprel_present: format.append('deprel')
        return format
    
    def extract(self,token_string,exceptions):
        '''
        Given a string representing a token, and a separator
        string which separates items in the token string,
        return a Token object with the attributes specified. 
        '''
        items = token_string.strip().split(self.separator)
        atts = dict((type,items[index]) for index,type in enumerate(self.format) if type not in ['ignore','fill'] and type not in exceptions and index<len(items))
        return Token(atts)
    
    def token_str(self,token):
        '''
        Given a single token object, return its string representation.
        '''
        items = []
        for f_type in self.format:
            if f_type!='ignore':
                item = token[f_type] if f_type!='fill' else '_'
                items.append(item if item else '_')
        return self.separator.join(items)
    
class TextReader(object):
    
    def __init__(self,formatfile=None,separator='\t'):
        '''
        Assign a Format object, so that the TextReader knows
        how to form Token objects from raw text.
        '''
        self.format = Format(separator,formatfile)
        self.add_ids = not self.format.id_present
        if not self.format.id_present:
            self.format.prepend_id()
        self.format.id_present = True
        
    def sentence_str(self,sentence):
        '''
        Given a list of token objects return its string representation.
        '''
        sentence_str = []
        for token in sentence:
            sentence_str.append(self.format.token_str(token))
        return '\n'.join(sentence_str)
        
    def sentences(self,infile,exceptions=None):
        '''
        Return a generator over sentences, where each sentence
        is a list of Token objects gleaned from a file using
        the assigned Format file.
        Exceptions should be a SET of relations which shouldn't be extracted
        into the Token objects. E.g. "deprel" and "head" in order to not
        read in the answers when already present.
        '''
        if exceptions is None: exceptions = set()
        with open(infile) as text:
            sentence = []
            for line in text:
                if line.strip():
                    sentence.append(self.format.extract("%s%s%s" % (len(sentence)+1,self.format.separator,line),exceptions) if self.add_ids 
                               else self.format.extract(line,exceptions))
                elif sentence:
                    yield sentence; sentence = []
            if sentence: yield sentence
    
    def sentence_from_list(self,stringlist):
        '''
        From a list of strings representing tokens, return
        a sentence, which is a list of Token objects.
        '''
        return [self.format.extract(token) for token in stringlist]
    
    @staticmethod
    def sentence_from_form_pos_tuple(token_list):
        full_tokens = []
        for i,token in enumerate(token_list):
            atts = {'form':token[0],
                    'pos' :token[1],
                    'id'  :str(i+1)}
            full_tokens.append(Token(atts))
        return full_tokens
        
if __name__ == "__main__":

    t = TextReader('/Volumes/LocalScratchHD/Local Home/adr27/EclipseProjects/workspace/ParsingSuite/parse_suite_repo/examples/format_minimal.txt')
    for i in t.sentences('/Volumes/LocalScratchHD/Local Home/adr27/EclipseProjects/workspace/ParsingSuite/parse_suite_repo/examples/sentence4.txt'):
        print t.sentence_str(i)
    pass
            
    
            