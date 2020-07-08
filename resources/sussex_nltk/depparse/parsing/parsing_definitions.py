'''
Created on 2 Jun 2012

@author: Andy
'''
from ..data_structures.parser_state import select_parser

def select_style(style):
    '''
    Call this function in order to select the correct Class
    representing the desired parsing style.
    '''
    if style == 'arc-eager':
        return ArcEagerParseStyle()
    elif style == 'arc-standard':
        return ArcStandardParseStyle()

class ParseStyle(object):
    '''
    Abstract class representing a parsing style.
    Empty functions should be overridden by subclass.
    
    Transition functions should give the option to
    perform the transition or not, and return True
    if the transition is possible else False.
    '''
    
    def __init__(self):
        pass
    
    def get_new_parse_state(self):
        '''
        Return a new parser state that is appropriate to this style of parsing
        '''
        return select_parser('1-stack')
    
    def new_training_sentence(self,sentence):
        '''
        Must be called for each new sentence that is being converted in the 
        training data. Performs any setup required.
        '''
        self.training_arcs,self.dep_counts = self._extract_arc_maps(sentence)
    
    def transition(self,state,name,label='',perform=True):
        '''
        Given a Parser state, the name of a transition and the
        label if required, determine if the specified transition is 
        possible. If perform==True, then also perform the transition.
        '''
        pass
    
    def optimum_training_transition(self,state):
        '''
        Given the parser state and the data gleaned from the training sentence
        during the call to self.new_training_sentence(), return a tuple of the
        form (transition_name,label) where label can be None.
        '''
        pass
    
    def _finished_deps(self,token):
        '''
        Return true if according to the data gleaned from the training sentence
        during the call to self.new_training_sentence(), the specified token has
        all of its dependents attached.
        '''
        arcs_needed = self.dep_counts[token['id']] if token['id'] in self.dep_counts else 0
        return token.deps == arcs_needed
    
    def _extract_arc_maps(self,sentence):
        '''
        Given a training sentence, collect those details which are necessary to 
        make decisions about what parsing actions should be performed at any given
        stage of the parse. This includes mapping pairs of token IDs to the label
        of the relation between them, and maintaining counts of how many dependents
        a token has.
        '''
        arcs = {}
        arc_counts = {}
        for token in sentence:
            head = token['head']
            arcs[(head,token['id'])] = token['deprel']
            if head in arc_counts:
                arc_counts[head]+=1
            else: arc_counts[head] = 1
        return arcs,arc_counts
    
    def _shift(self,state,perform=True):
        if state.buffer:
            if perform:
                state.stack.append(state.buffer.popleft())
            return True
        else: return False
    
class ArcStandardParseStyle(ParseStyle):
    
    def __init__(self):
        super(ArcStandardParseStyle,self).__init__()
        
    def transition(self,state,name,label=None,perform=True):
        if name=='left':
            assert label is not None, "Label shouldn't be None"
            return self._left_arc(state, label, perform)
        elif name=='right':
            assert label is not None, "Label shouldn't be None"
            return self._right_arc_pop(state, label, perform)
        elif name=='shift':
            return self._shift(state, perform)
        
    def optimum_training_transition(self,state):
        if state.stack:
            shead = state.stack[-1]['id']
            bhead = state.buffer[0]['id']
            if (bhead,shead) in self.training_arcs:
                label = self.training_arcs[(bhead,shead)]
                self._left_arc(state, label)
                return ('left',label)
            elif (shead,bhead) in self.training_arcs and self._finished_deps(state.buffer[0]):
                label = self.training_arcs[(shead,bhead)]
                self._right_arc_pop(state, label)
                return ('right',label)
            else:
                self._shift(state)
                return ('shift',None)
        else:
            self._shift(state)
            return ('shift',None) 
        
    def _left_arc(self,state,label,perform=True):
        if state.stack and state.buffer and not state.stack[-1].is_root():
            if perform:
                dependent = state.stack.pop()
                head = state.buffer[0]
                state.add_arc(head,dependent,label)
            return True
        else: return False
        
    def _right_arc_pop(self,state,label,perform=True):
        if state.stack and state.buffer:
            if perform:
                head = state.stack.pop()
                dependent = self.buffer.popleft()
                state.buffer.appendleft(head)
                state.add_arc(head,dependent,label)
            return True
        else: return False
    
class ArcEagerParseStyle(ParseStyle):
    
    def __init__(self):
        super(ArcEagerParseStyle,self).__init__()
        
    def transition(self,state,name,label=None,perform=True):
        if name=='left':
            assert label is not None,"label shouldn't be None"
            return self._left_arc_pc(state, label, perform)
        elif name=='right':
            assert label is not None,"label shouldn't be None"
            return self._right_arc_push(state, label, perform)
        elif name=='shift':
            return self._shift(state, perform)
        elif name=='reduce':
            return self._reduce(state, perform)
    
    def optimum_training_transition(self,state):
        if state.stack:
            shead = state.stack[-1]['id']
            bhead = state.buffer[0]['id']
            if (bhead,shead) in self.training_arcs:
                label = self.training_arcs[(bhead,shead)]
                self._left_arc_pc(state,label)
                return ('left',label)
            elif (shead,bhead) in self.training_arcs:
                label = self.training_arcs[(shead,bhead)]
                self._right_arc_push(state,label)
                return ('right',label)
            elif self._finished_deps(state.stack[-1]) and state.stack[-1].parent:
                self._reduce(state)
                return ('reduce',None)
            else:
                self._shift(state)
                return ('shift',None)
        else:
            self._shift(state)
            return ('shift',None)
        
    def _reduce(self,state,perform=True):
        if state.stack and state.stack[-1].parent:
            if perform:
                state.stack.pop()
            return True
        else: return False
        
    def _right_arc_push(self,state,label,perform=True):
        if state.stack and state.buffer:
            if perform:
                head = state.stack[-1]
                dependent = state.buffer.popleft()
                state.stack.append(dependent)
                state.add_arc(head,dependent,label)
            return True
        else: return False
        
    def _left_arc_pc(self,state,label,perform=True):
        if state.stack and state.buffer and not state.stack[-1].is_root() and not state.stack[-1].parent:
            if perform:
                dependent = state.stack.pop()
                head = state.buffer[0]
                state.add_arc(head,dependent,label)
            return True
        else: return False
    