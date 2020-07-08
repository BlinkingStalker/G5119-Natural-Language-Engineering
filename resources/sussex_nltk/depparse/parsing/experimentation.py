'''
Created on Jul 5, 2012

@author: adr27
'''
import os
from parsing_functions import DependencyParser
from ..data_handling.configuration_handling import Configuration
from ..data_handling.text_handling import TextReader

class ParsedSentence(object):
    def __init__(self,sentence):
        self.tokens = [BasicToken(token) for token in sentence]
    def __iter__(self):
        return iter(self.tokens)
    def __getitem__(self,index):
        return self.tokens[index]
    def find_all_dependants(self,words):
        words = set(words)
        ids = set()
        for token in self.tokens:
            if token.form in words:
                ids.add(token.id)
        for token in self.tokens:
            if token.head in ids:
                yield token
    def raw(self):
        return ' '.join([token.form for token in self.tokens])
    def __str__(self):
        return '\n'.join([str(token) for token in self.tokens])

    def get_query_tokens(self, form):
        return [token for token in self.tokens if token.form==form]
    def get_head(self, token):
        return BasicToken() if token.head == 0 else self.tokens[token.head-1]

    def get_dependants(self, token):
        return [t for t in self.tokens if t.head == token.id]

    def get_token_by_id(self, token_id):
        return self.tokens[token_id-1] if 0 < token_id <= len(self.tokens) else BasicToken()
    def get_dependants_by_head_id(self, head_id):
        return [token for token in self.tokens if token.head==head_id]

class BasicToken(object):
    def __init__(self,token=None):
        if token is not None:
            self.id  = int(token['id'])
            self.form = token['form']
            self.pos = token['pos']
            self.head = int(token['head'])
            self.deprel = token['deprel']
        else:
            self.id = 0
            self.form = "root"
            self.pos = "root"
            self.head = 0
            self.deprel = "root"

    def __str__(self):
        return "%s\t%s\t%s\t%s\t%s" % (self.id,self.form,self.pos,self.head,self.deprel)

def parse_sentences(sentences,feature_indices_file,classifier_model):
    d = DependencyParser()
    d.setup_parser_for_sentence_object_parsing(feature_indices_file,classifier_model)
    for sentence in sentences:
        formatted_sent = TextReader.sentence_from_form_pos_tuple(sentence)
        yield ParsedSentence(d.parse_sentence(formatted_sent))

def get_parser(feature_indices_file,classifier_model):
    d = DependencyParser()
    d.setup_parser_for_sentence_object_parsing(feature_indices_file,classifier_model)
    return d

def parse_with_parser(parser, sentences):
    for sentence in sentences:
        formatted_sent = TextReader.sentence_from_form_pos_tuple(sentence)
        yield ParsedSentence(parser.parse_sentence(formatted_sent))
