#Functionality from week3 to be imported into future notebooks
#Update this and add your own functions too

#----------------------------------------------------
#Imports
import sys
#sys.path.append(r'T:\Departments\Informatics\LanguageEngineering') 
sys.path.append(r'\\ad.susx.ac.uk\ITS\TeachingResources\Departments\Informatics\LanguageEngineering\resources')
sys.path.append(r'/Users/juliewe/resources')

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
#----------------------------------------------------
#Lab_3_1
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
 
#Create an Amazon corpus reader pointing at only dvd reviews
dvd_reader = AmazonReviewCorpusReader().category("dvd")

#The following two lines use the documents function on the Amazon corpus reader. 
#This returns a generator over reviews in the corpus. 
#Each review is an instance of a Python class called AmazonReview. 
#An AmazonReview object contains all the data about a review.
pos_train, pos_test = split_data(dvd_reader.positive().documents())
neg_train, neg_test = split_data(dvd_reader.negative().documents())

#You can also combine the training data
train = pos_train + neg_train
    
    
def get_all_words(amazon_reviews):
    return reduce(lambda words,review: words + review.words(), amazon_reviews, [])

class SimpleClassifier(ClassifierI): 

    def __init__(self, pos, neg):
        self._labels=["P","N"]
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

    def classify_many(self, docs): 
        return [self.classify(doc.words() if hasattr(doc, 'words') else doc) for doc in docs] 

    def labels(self): 
        return self._labels
    
    
def most_frequent_words(freqdist,k):
    return [word for word,count in freqdist.most_common(k)]    
    
class SimpleClassifier_mf(SimpleClassifier):
    
    def __init__(self,k):
        self.k=k
    
    def train(self,pos_train,neg_train):
        pos_freqdist = FreqDist(get_all_words(pos_train))
        neg_freqdist = FreqDist(get_all_words(neg_train))
        self._pos=most_frequent_words(pos_freqdist,self.k)
        self._neg=most_frequent_words(neg_freqdist,self.k)
        

#------------------------
#Lab_3_2
#put your NB classifier class code here!
class NBClassifier(ClassifierI):
    
    def __init__(self):
        self._labels=["P","N"]
        pass
    
    def labels(self): 
        return self._labels
    
    def set_known_vocabulary(self,training_data):
        known=[]
        for doc,label in training_data:
            known+=list(doc.keys())
        self.known= set(known)
    
    def set_priors(self,training_data):
        priors={}
        for (doc,label) in training_data:
            priors[label]=priors.get(label,0)+1
        total=sum(priors.values())
        for key,value in priors.items():
            priors[key]=value/total
        self.priors=priors
        
    def set_cond_probs(self,training_data):
        conds={}
        for(doc,label) in training_data:
            classcond=conds.get(label,{})
            for word in doc.keys():
                classcond[word]=classcond.get(word,0)+1
        
            conds[label]=classcond
    
        for label, classcond in conds.items():
            for word in self.known:
        
                classcond[word]=classcond.get(word,0)+1
            conds[label]=classcond
            
        for label,dist in conds.items():
            total=sum(dist.values())
            conds[label]={key:value/total for (key,value) in dist.items()}
        
        self.conds=conds
    
    def train(self,training_data):
        self.set_known_vocabulary(training_data)
        self.set_priors(training_data)
        self.set_cond_probs(training_data)
    
    def classify(self,doc):
        doc_probs={key:math.log(value) for (key,value) in self.priors.items()}
        for word in doc.keys():
            if word in self.known:
                doc_probs={classlabel:sofar+math.log(self.conds[classlabel].get(word,0)) for (classlabel,sofar) in doc_probs.items()}

        highprob=max(doc_probs.values())
        classes=[c for c in doc_probs.keys() if doc_probs[c]==highprob]
        return random.choice(classes)
    
    
    
