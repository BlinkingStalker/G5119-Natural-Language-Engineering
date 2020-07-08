'''
Created on 4 May 2012

@author: Andrew D. Robertson
'''

from ..data_structures.mapping import BiMap

class FeatureIndexer(object):
    '''
    Class for keeping track of unique IDs for hashable types.
    '''
    
    def __init__(self):
        '''Initialise'''
        self._bimap = BiMap()
        self._id = 1
        
    def get_id(self,item):
        '''
        Given some hashable item, look it up in the two-way
        map. If it is present, return its ID. Otherwise create
        a new ID and store it with the item. Return the ID.    
        '''
        if item in self._bimap:
            return self._bimap.value_of(item)
        else:
            self._bimap[item] = self._id
            self._id += 1
            return self._id-1
        
    def sorted_ids(self):
        return sorted(self._bimap)
    def reverse_sorted_ids(self):
        return sorted(((id,name) for name,id in self._bimap))
        
    def get_value(self,id):
        '''
        Given the unique ID return None if that ID hasn't been assigned
        else return the value mapped to that ID.
        '''
        return self._bimap.key_of(id)
        
    def get_key(self):
        '''
        Return a string providing a key. This will be a list
        of unique IDs and their corresponding hashable items.
        '''
        out = []
        for k,v in self._bimap:
            out.append('%s %s'%(v,k))
        return '\n'.join(out)
    
    def __len__(self):
        return len(self._bimap)

class FeatureLine(object):
    '''
    Class represents a line in the feature specification file
    '''
    
    def __init__(self,addresses,featuretypes,feature_manager):
        self.addresses = addresses
        self._ftypes = featuretypes
        self._fm = feature_manager

    
    def _type_str(self,ftype,args):
        if args is None:
            return ftype
        else: return "%s:(%s)"%(ftype,','.join(args))
    
class SingleAddressFeatureLine(FeatureLine):
    '''
    Class represents a line in the feature specification file, where
    the features specified operate over a single token.
    '''
    
    def __init__(self,address,featuretypes,feature_manager):
        super(SingleAddressFeatureLine, self).__init__(address,featuretypes,feature_manager)
        
    def eval_features(self,token):
        '''
        Return a generator over the values of the requested 
        feature types in the following format: (feature_type,feature_value)
        '''
        if token:
            return ((self._type_str(ftype,args),str(token.get_feature(ftype,args)) if token.get_feature(ftype,args) else self._fm.no_feature) for ftype,args in self._ftypes)
        else:
            return ((self._type_str(ftype,args),self._fm.no_token) for ftype,args in self._ftypes)
      
    
class MultiAddressFeatureLine(FeatureLine):
    '''
    Class represents a line in the feature specification file, where
    the features specified operate over a multiple tokens.
    '''
    
    def __init__(self,addresses,featuretypes,feature_manager):
        super(MultiAddressFeatureLine, self).__init__(addresses,featuretypes,feature_manager)
        
    def eval_featuresOld(self,tokens):
        if not any(tokens): return
        pairwise = True if len(tokens) == 2 else False
        for type,args in self._ftypes:
            if pairwise:
                try:
                    if type == '+dist':
                        d = self._fm.dist_excl(tokens[0],tokens[1],args)
                        yield (self._type_str(type,args),str(d)); continue
                    elif type == '-dist':
                        d = self._fm.dist_incl(tokens[0],tokens[1],args)
                        yield (self._type_str(type,args),str(d)); continue
                    elif type == 'dist':
                        d = tokens[0].dist(tokens[1])
                        yield (self._type_str(type,args),str(d)); continue
                except (AttributeError,TypeError): yield (self._type_str(type,args),self._fm.no_token)
            if type == 'join':
                f = []
                for token,arg in zip(tokens,args):
                    try: f.append(str(token.get_feature(arg))if token.get_feature(arg) else self._fm.no_feature)
                    except AttributeError: f.append(self._fm.no_token)
                yield (self._type_str(type, args),'|'.join(f))
                
    def eval_features(self,tokens):
        pairwise = True if len(tokens) == 2 else False
        for type,args in self._ftypes:
            if pairwise:
                try:
                    if type == '+dist':
                        d = self._fm.dist_excl(tokens[0],tokens[1],args)
                        yield (self._type_str(type,args),str(d)); continue
                    elif type == '-dist':
                        d = self._fm.dist_incl(tokens[0],tokens[1],args)
                        yield (self._type_str(type,args),str(d)); continue
                    elif type == 'dist':
                        d = tokens[0].dist(tokens[1])
                        yield (self._type_str(type,args),str(d)); continue
                except (AttributeError,TypeError): 
                    assert None in tokens, "Houston, we have a problem."
                    yield (self._type_str(type,args),self._fm.no_token)
            if type == 'join':
                f = []
                for token,arg in zip(tokens,args):
                    try: f.append(str(token.get_feature(arg)) if token.get_feature(arg) else self._fm.no_feature)
                    except AttributeError: f.append(self._fm.no_token)
                yield (self._type_str(type, args),'|'.join(f))        

class FeatureManager(object):
    '''
    Handles reading in feature model specifications, 
    and tracking of new features.
    '''

    def __init__(self,feature_model_file):
        self.features = self._read_feature_model(feature_model_file)
        self._fi = FeatureIndexer()
        self._ti = FeatureIndexer()
        self.base_trans = set()
        self._cumul = False
        self.set_null_feature_handling('<no_value>', '<no_value>')
        
    def init_sentence(self,sent):
        if self._cumul: self._cumul_init(sent)
        
    def set_null_feature_handling(self,no_feature='<no_value>',no_token='<no_token>'):
        self.no_feature = no_feature
        self.no_token = no_token
    
    def get_id(self,address,featuretype,value):
        '''Get the ID of a feature and its value. If unseen, then generate new ID'''
        return self._fi.get_id('|'.join([self._str_add(address),featuretype,str(value)]))
    def get_class_id(self,transition,label=None):
        '''Get the ID of a transition and its label. If unseen then generate new ID'''
        self.base_trans.add(transition)
        return self._ti.get_id('|'.join([transition,label])) if label else self._ti.get_id(transition)
    
    def get_transition(self,id):
        '''Given an ID return transition string'''
        return self._ti.get_value(id)
    def get_transition_tuple(self,id):
        '''Given an ID return transition and its label in a tuple'''
        transition = self._ti.get_value(id).split('|')
        return (transition[0],transition[1]) if len(transition)==2 else (transition[0],None)
    
    def _str_add(self,address):
        '''Get the string form of an address or multiple addresses'''
        if isinstance(address[0],basestring):
            return '<'.join(address)
        else:
            return ','.join(['<'.join(ad) for ad in address])
    
    def _read_feature_model(self,infile):
        '''
        Given a file specifying what features that parser should
        use, return a list of FeatureLine objects, containing
        this information.
        '''
        with open(infile) as ftable:
            return [self._build_feature(line) for line in ftable 
                                              if line.strip() and not line.startswith('#')]  
    def _build_feature(self,featureline):
        '''
        Given a line in the feature specification file, return an object representing 
        the line, with information such as the tokens over which the feature operates,
        and what the feature is.
        '''
        items = featureline.split(':')
        addresses = [self._build_address(add) for add in items[0].split()]
        del items[0]
        functions = [self._build_function(func.strip()) for func in items]
        if len(addresses) > 1:
            return MultiAddressFeatureLine(addresses,functions,self)
        else:
            return SingleAddressFeatureLine(addresses,functions,self)
                
    def _build_function(self,func_string):
        if (func_string[0] == '+') or (func_string[0]=='-'): self._cumul=True
        try:
            pstart = func_string.index('(')
            return (func_string[:pstart],func_string[pstart+1:-1].split(','))
        except ValueError:
            return (func_string,None)
            
    def _build_address(self,add_string):
        cmpnts = add_string.split('(')
        loc,index = cmpnts[-1].split('[')
        cmpnts[-1] = loc
        cmpnts.append(index.split(']')[0])
        return cmpnts 
    
    def _cumul_init(self,sent):
        '''
        Creates a dictionary that maps pos tags to their cumulative frequency
        in the sentence. Used for distance calculations that are pos dependent.
        '''
        self._SENTLENGTH = len(sent)+1
        self._cumulpos = {}
        for token in sent:
            pos = token.get_feature('pos')
            if pos not in self._cumulpos:
                self._cumulpos[pos] = [0]*(len(sent)+1)
        self._cumulpos['root'] = [1]*(len(sent)+1)
        for i in xrange(len(sent)):
            for pos,cumul in self._cumulpos.iteritems():
                postag = sent[i].get_feature('pos')
                cumul[i+1] = cumul[i]+1 if pos==postag else cumul[i]
                
    def _pos_distance(self,pos,i,j):
        '''
        Return the distance between two token IDs in terms 
        of a single postag 'pos'
        '''
        if pos not in self._cumulpos:
            self._cumulpos[pos] = [0]*self._SENTLENGTH
        if j > i:
            return self._cumulpos[pos][j-1]-self._cumulpos[pos][i]
        else:
            return self._cumulpos[pos][i-1]-self._cumulpos[pos][j]
        
    def dist_incl(self,t1,t2,poslist):
        '''
        Find the distance between t1 and t2 in terms of the postags
        in 'poslist'
        '''
        dist = 0
        for pos in poslist:
            if   pos == 'pos1': pos = t1.get_feature('pos')
            elif pos == 'pos2': pos = t2.get_feature('pos') 
            dist+=self._pos_distance(pos,t1['id'],t2['id'])
        return dist
    
    def dist_excl(self,t1,t2,poslist):
        '''
        Find the distance between t1 and t2 in terms of the postags
        which do not appear in 'poslist'
        '''
        dist = 0
        poslist = set(poslist)
        try: 
            poslist.remove('pos1')
            poslist.add(t1.get_feature('pos'))
        except KeyError: pass
        try:
            poslist.remove('pos2')
            poslist.add(t2.get_feature('pos'))
        except KeyError: pass
        for pos in self._cumulpos:
            if pos not in poslist:
                dist+=self._pos_distance(pos,t1['id'],t2['id'])
        return dist
    
if __name__ == '__main__':
#    infile = '/Users/Andy/EclipseProjects/workspace/ParsingSuite/parse_suite_repo/examples/features.txt'
#    infile = '/Volumes/LocalScratchHD/Local Home/adr27/EclipseProjects/workspace/ParsingSuite/parse_suite_repo/examples/features.txt'
#    fm = FeatureManager(infile)
#    print 'yes'
    pass
        