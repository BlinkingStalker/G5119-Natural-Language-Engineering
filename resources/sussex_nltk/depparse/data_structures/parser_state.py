'''
Created on 1 Jun 2012

@author: Andrew D. Robertson
'''
from collections import deque

from ..data_handling.text_handling import Token

def select_parser(style):
    if style=='1-stack':
        return OneStackParserState()

class OneStackParserState(object):

    def __init__(self,buffer=None,stack=None,arcs=None):
        self.buffer = buffer if buffer else deque()
        self.stack  = stack  if stack  else []
        self.arcs   = arcs   if arcs   else {}
    
    def initialise(self,sentence):
        '''
        Initialise the parser state with a new sentence
        '''
        self.buffer = deque(sentence)
        root = Token({'id':'0',
                      'form': 'ROOT'})
        self.stack = [root]
        self.arcs = {}
        
    def is_terminal(self):
        '''Return true if the parser is in a terminal state'''
        return not self.buffer
    
    def add_arc(self,head,dependent,label):
        '''Add a dependency relation from head to dependent of type label'''
        self.arcs[(head['id'],dependent['id'])] = label
        dependent.set_head(head,label)
        
    def __str__(self):
        return '\n'.join(['Stack:\t%s' %(','.join(str(token) for token in self.stack)),\
                          'Buffer:\t%s' %(','.join(str(word) for word in self.buffer)),\
                          'Arcs:\t%s' %(','.join(['->'.join([str(head),rel,str(dep)]) for (head,dep),rel in self.arcs.iteritems()]))])
        
        
if __name__ == '__main__':
    pass