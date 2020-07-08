'''
Created on 8 Mar 2011

Module implementing map data structures.

@author: Andew D. Robertson
'''

class BiMap(object):
    '''
    Class implementing a two way hashmap.
    '''
    
    def __init__(self,keys=None,values=None):
        self._orientation = True
        self._k2v = {}
        self._v2k = {}
        if keys and values:
            for k,v in zip(keys,values):
                self._k2v[k] = v
                self._v2k[v] = k
    
    def reverse(self): self._orientation = not self._orientation
    def forward(self): self._orientation = True
    def backward(self): self._orientation = False
    
    def value_of(self,key):
        return self._k2v[key] if key in self._k2v else None
    
    def key_of(self,value):
        return self._v2k[value] if value in self._v2k else None
    
    def __contains__(self,key):
        return key in self._k2v
    
    def __getitem__(self,index):
        return self._k2v[index] if self._orientation else self._v2k[index]
    
    def __setitem__(self,key,value):
        self._k2v[key] = value
        self._v2k[value] = key
        
    def __iter__(self):
        return self._k2v.iteritems() if self._orientation else self._v2k.iteritems()
    
    def __str__(self):
        return '\n'.join(':'.join([str(k),str(v)]) for k,v in self)
    
    def __len__(self):
        return len(self._k2v)

if __name__ == '__main__':
    pass