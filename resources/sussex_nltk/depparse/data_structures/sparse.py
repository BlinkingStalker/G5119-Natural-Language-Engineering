'''
Created on 17 Feb 2011

Module implementing sparse data structures.

@author: Andrew D. Robertson
'''
from bisect import bisect
from numpy import argsort,take
from math import sqrt


class SparseVector(object):
    
    
    def __init__(self,elements=None,indices=None,features_ordered=False,always_add_at_end=False):
        self._elements = []
        self._indices = []
        self._naive = always_add_at_end
        if elements and indices:
            self._elements += elements
            self._indices += indices
            if not features_ordered:
                ids = argsort(self._indices)
                self._indices = list(take(self._indices,ids))
                self._elements = list(take(self._elements,ids))
        
    def __setitem__(self,key,value):
        if self._naive:
            self.add_at_end(key,value)
        else:
            i = bisect(self._indices,key)
            self._indices.insert(i, key)
            self._elements.insert(i, value)
            
    def binary_set(self,key):
        self[key] = 1
    
    def __iter__(self):
        return ((index,item) for index,item in zip(self._indices,self._elements))
    
    def __len__(self):
        return len(self._elements)
        
    def add_at_end(self,key,value):
        self._elements.append(value)
        self._indices.append(key)
        
    def __str__(self):
        return ' '.join([':'.join([str(index),str(item)]) for index,item in zip(self._indices,self._elements)])
    
    def vector_dict(self):
        out = {}
        for index,item in zip(self._indices,self._elements):
            out[index] = item
        return out
    
class DictSparseVector(object):
    
    def __init__(self,elements=None,indices=None,items=None):
        self.items = items if items else {}
        if elements and indices:
            for i,e in zip(indices,elements):
                self.items[i] = e
                
    def __setitem__(self,key,value):
        self.items[key] = value
        
    def __getitem__(self,key):
        return self.items[key]
        
    def vector_dict(self):
        return self.items
        
    def binary_set(self,key):
        self[key] = 1
        
    def sorted_str(self):
        return ' '.join([':'.join([str(index),str(item)]) for index,item in sorted(self.items.iteritems())])
    
    def normalise(self):
        length = float(sqrt(sum(x**2 for x in self.items.itervalues())))
        for i in self.items:
            self.items[i]/=length
        
    def binary_normalise(self):
        length = float(sqrt(len(self.items)))
        for i in self.items:
            self.items[i]/=length
                
    def __str__(self):
        return ' '.join([':'.join([str(index),str(item)]) for index,item in self.items.iteritems()])
        
                  
if __name__ == '__main__':
    pass