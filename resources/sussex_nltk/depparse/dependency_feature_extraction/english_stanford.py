'''
Created on Oct 18, 2012

@author: adr27
'''
import itertools

root_value = 'root'

class Item(object):
    '''
    Class representing a single token in a dependency tree
    '''
    def __init__(self,id,form,pos,head,deprel):
        self.id = int(id)
        self.form = form
        self.pos = pos
        self.head = int(head)
        self.deprel = deprel
        
    @staticmethod
    def root_item():
        return Item(0,root_value,None,None,None)
        
class Entry(object):
    '''
    Class representing tokens grouped in some fashion.
    Initialised with starting token. Initial sentence is
    one token per entry.
    '''
    def __init__(self,item):
        self.items = [item]
        self.u_head = item.head
        self.u_deprel = item.deprel
        
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self,key):
        return self.items[key]
    
    def __iter__(self):
        return (item.form for item in self.items)
        
    def add(self,item):
        self.items.append(item)

class FeatureProcessor(object):
    
    def __init__(self,feature_writer):
        self.out = feature_writer
        
    def initialise_sentence(self,sentence):
        '''
        *sentence* should be a list of tokens, where
        each token is a list of string attributes:
        ID, word form, PoS tag, ID of head, deprel
        '''
        return [Item(*token) for token in sentence]
    
    def merge_conjuncts(self,sentence):
        '''
        Sentence should be a list of Item instances
        '''
        merged = {0: Entry(Item.root_item())}
        for token in sentence:
            if 'conj' in token.deprel and token.id not in merged:
                if token.head not in merged:
                    raise IndexError("Right chaining conjuncts, re-implementation required.")
                merged[token.head].add(token)
                merged[token.id] = merged[token.head]
            elif token.id not in merged:
                merged[token.id] = Entry(token)
        return merged   
    
    def permute_conjunctions(self,entry):
        for token1,token2 in itertools.permutations(entry,2):
            self.out.add_relation(token1,'conj',token2)
    
    def permute_coordinated_relations(self,head_entry,neighbour_entry):
        self.permute_conjunctions(head_entry)
        self.permute_all_pair_relations(head_entry, head_entry.u_deprel, neighbour_entry)
                
    def permute_all_pair_relations(self,head_entry,relation,neighbour_entry):
        for head_token in head_entry:
            for neighbour_token in neighbour_entry:
                self.out.add_relation(head_token,relation,neighbour_token)

class AdverbFeatureExtractor(object):
    
    def __init__(self,feature_writer):
        self.processor = FeatureProcessor(feature_writer)
        
    def extract_features(self,sentence):
        sentence = self.processor.initialise_sentence(sentence)
        merged   = self.processor.merge_conjuncts(sentence)
        for token in sentence:
            if 'RB' in token.pos and 'conj' not in token.deprel:
                self.processor.permute_coordinated_relations(merged[token.id],merged[token.head])
    
class NounFeatureExtractor(object):
    
    def __init__(self,feature_writer):
        self.processor = FeatureProcessor(feature_writer)
        
    def extract_features(self,sentence):
        sentence = self.processor.initialise_sentence(sentence)
        merged   = self.processor.merge_conjuncts(sentence)
        for token in sentence:
            pass
        
class VerbFeatureExtractor(object):
    
    def __init__(self,feature_writer,accepted_mods=None):
        self.processor = FeatureProcessor(feature_writer)
        self.accepted_mods = accepted_mods if accepted_mods else set(['RB','NN','NP','VB','MD','IN','JJ'])
        
    def extract_features(self,sentence,mod_rels=True):
        sentence = self.processor.initialise_sentence(sentence)
        merged   = self.processor.merge_conjuncts(sentence)
        for token in sentence:
            if 'conj' not in token.deprel:
                if 'VB' in token.pos:
                    self.processor.permute_coordinated_relations(merged[token.id],merged[token.head])
                elif mod_rels and 'VB' in sentence[token.head].pos:
                    tag = token.pos[:2] if len(token.pos)>2 else token.pos
                    if tag in self.accepted_mods:
                        self.processor.permute_all_pair_relations(merged[token.head], token.deprel, merged[token.id])
                    

class AdjectiveFeatureExtractor(object):
    
    def __init__(self,feature_writer):
        self.processor = FeatureProcessor(feature_writer)
        
    def extract_features(self,sentence):
        sentence = self.processor.initialise_sentence(sentence)
        merged   = self.processor.merge_conjuncts(sentence)
        for token in sentence:
            if 'JJ' in token.pos and 'conj' not in token.deprel:
                self.processor.permute_coordinated_relations(merged[token.id],merged[token.head])
    
    
if __name__ == "__main__":
    pass