#Imports
import sys
#sys.path.append(r'T:\Departments\Informatics\LanguageEngineering') 
sys.path.append(r'\\ad.susx.ac.uk\ITS\TeachingResources\Departments\Informatics\LanguageEngineering\resources')
sys.path.append(r'/Users/juliewe/Documents/teaching/NLE2018/resources')

import re
import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest
from nltk.tokenize import word_tokenize
from sussex_nltk.corpus_readers import ReutersCorpusReader
from nltk.probability import FreqDist # see http://www.nltk.org/api/nltk.html#module-nltk.probability
from sussex_nltk.corpus_readers import AmazonReviewCorpusReader
from functools import reduce # see https://docs.python.org/3/library/functools.html
from nltk.classify.api import ClassifierI
import random
from random import sample
import math
from nltk.stem.wordnet import WordNetLemmatizer

#-------------

def split_data(data, ratio=0.7): # when the second argument is not given, it defaults to 0.7
    """
    Given corpus generator and ratio:
     - partitions the corpus into training data and test data, where the proportion in train is ratio,

    :param data: A corpus generator.
    :param ratio: The proportion of training documents (default 0.7)
    :return: a pair (tuple) of lists where the first element of the 
            pair is a list of the training data and the second is a list of the test data.
    """
    
    data = list(data) # data is a generator, so this puts all the generated items in a list
 
    n = len(data)  #Found out number of samples present
    train_indices = sample(range(n), int(n * ratio))          #Randomly select training indices
    test_indices = list(set(range(n)) - set(train_indices))   #Other items are testing indices
 
    train = [data[i] for i in train_indices]           #Use training indices to select data
    test = [data[i] for i in test_indices]             #Use testing indices to select data
 
    return (train, test)                       #Return split data
 

def normalise(word):
    lemma=WordNetLemmatizer()
    return lemma.lemmatize(word.lower())
    
def feature_extract(review):
    #print(review.words())
    return {normalise(word):True for word in review.words()}

def get_training_test_data(cat,ratio=0.7):
    """
    Given a category of review and a ratio, make appropriate partitions of positive and negative reviews from that category
    :param cat: A String specifying the category of review e.g., "dvd"
    :param ratio: A float specifying the proportion of training documents, default value = 0.7
    :return: A pair of lists where first element is the training set and second element is the testing set
    """
    reader=AmazonReviewCorpusReader().category(cat)
    pos_train, pos_test = split_data(reader.positive().documents(),ratio=ratio)
    neg_train, neg_test = split_data(reader.negative().documents(),ratio=ratio)
    train_data=[(review,'P') for review in pos_train]+[(review,'N') for review in neg_train]
    test_data=[(review,'P') for review in pos_test]+[(review,'N') for review in neg_test]
    return train_data,test_data
    
def get_all_words(amazon_reviews):
    return reduce(lambda words,review: words + review.words(), amazon_reviews, [])

class SimpleClassifier(ClassifierI): 

    def __init__(self, pos, neg): 
        self._pos = pos 
        self._neg = neg 

    def classify(self, words): 
        score = 0
        
        for word in words:
            if word in self._pos:
                score+=1
            if word in self._neg:
                score-=1
        
        return "N" if score < 0 else "P" 

    def batch_classify(self, docs): 
        return [self.classify(doc.words() if hasattr(doc, 'words') else doc) for doc in docs] 

    def labels(self): 
        return ("P", "N")
    
class ConfusionMatrix:
    def __init__(self,predictions,goldstandard,classes=("P","N")):
        (self.c1,self.c2)=classes
        self.TP=0
        self.FP=0
        self.FN=0
        self.TN=0
        for p,g in zip(predictions,goldstandard):
            if g==self.c1:
                if p==self.c1:
                    self.TP+=1
                else:
                    self.FN+=1
            
            elif p==self.c1:
                self.FP+=1
            else:
                self.TN+=1
        
    
    def precision(self):
        p=self.TP/(self.FP+self.TP)   
        return p
    
    def recall(self):
        r=self.TP/(self.TP+self.FN)
        
        return r
    
    def f1(self):
        p=self.precision()
        r=self.recall()
        f1=2*p*r/(p+r)
        
        return f1 
    def accuracy(self):
        a=(self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)
        return a
    def error(self):
        return 1-self.accuracy()
    
    def get_measure(self,measure):
        if measure=="precision":
            return self.precision()
        elif measure=="recall":
            return self.recall()
        elif measure=="f1":
            return self.f1()
        elif measure=="accuracy":
            return self.accuracy()
        elif measure=="error":
            return self.error()
    
def classifier_evaluate(classifier,test_data,measures):
    
    docs,goldstandard=zip(*test_data) #note this neat pythonic way of turning a list of pairs into a pair of lists
    predictions=classifier.batch_classify(docs)
    confusion=ConfusionMatrix(predictions,goldstandard)
    
    return [confusion.get_measure(m) for m in measures]