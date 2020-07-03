#Selected solutions to programming exercises for NLE Week 2
#import it into a Jupyter notebook using:
#from W2solutions.py import *
#then call test_solution(question) to run and display output

#-----------------------------------
#"2_1, 2.4" (DIY Tokenizer)

import re    #import regex module

def tokenise(sentence):
    sentence = re.sub("'(s|m|(re)|(ve)|(ll)|(d))\s", " '\g<1> ",sentence + " ")
    sentence = re.sub("s'\s", "s ' ",sentence)
    sentence = re.sub("n't\s", " n't ",sentence)
    sentence = re.sub("gonna", "gon na",sentence)
    sentence = re.sub("\"(.+?)\"", "`` \g<1> ''",sentence)   
    sentence = re.sub("([.,?!])", " \g<1> ", sentence)
    return sentence.split()

testsentence = "After saying \"I won't help, I'm gonna leave!\", on his parents' arrival, the boy's behaviour improved."


#-----------------------------------
#"2_2, 1.1" (Number normalization)
import re

def normalise(tokenlist):
    tokenlist=[token.lower() for token in tokenlist]
    tokenlist=["NUM" if token.isdigit() else token for token in tokenlist]
    tokenlist=["Nth" if (token.endswith(("nd","st","th")) and token[:-2].isdigit()) else token for token in tokenlist]
    tokenlist=["NUM" if re.search("^[+-]?[0-9]+\.[0-9]",token) else token for token in tokenlist]
    return tokenlist

tokens = ["The", "1st", "and", "2nd", "placed", "runners", "lapped", "the", "5th","."]

#---------------------------------

def test_solution(question):
    if question=="2_1, 2.4":
        print(tokenise(testsentence))
    elif question=="2_2, 1.1":
        print(normalise(tokens))
    else:
        print("No solution for {}".format(question))
        
        
        