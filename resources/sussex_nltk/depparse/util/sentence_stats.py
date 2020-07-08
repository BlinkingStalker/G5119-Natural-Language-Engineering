'''
Created on Oct 18, 2012

@author: adr27
'''
import os

def count_sentences_in_file(inputfile):
    count = 0
    in_sentence = False
    with open(inputfile) as infile:
        for line in infile:
            if line.strip():
                in_sentence = True
            elif in_sentence:
                count+=1; in_sentence = False
    return count

def count_sentences_in_dir(inputdir,filter=None):
    count = 0
    for filename in os.listdir(inputdir):
        if not filename.startswith('.'):
            if filter and filter in filename:
                filepath = os.path.join(inputdir,filename)
                count += count_sentences_in_file(filepath)
    return count

def count_sentences_with_rel_in_file(inputfile,rel,rel_index=7,separator='\t'):
    rel_present = False
    in_sentence = False
    count = 0
    with open(inputfile) as infile:
        for line in infile:
            if line.strip():
                in_sentence = True
                if rel in line.split(separator)[rel_index]:
                    rel_present = True
            elif in_sentence:
                if rel_present:
                    count+=1; rel_present = False
                in_sentence = False
    return count
                
def count_sentences_with_rel_in_dir(inputdir,rel,rel_index=7,separator='\t',filter=None):
    count = 0
    for filename in os.listdir(inputdir):
        if not filename.startswith('.'):
            if filter and filter in filename:
                filepath = os.path.join(inputdir,filename)
                count += count_sentences_with_rel_in_file(filepath,rel,rel_index,separator)
    return count


if __name__ == "__main__":
    pass
      
    inputdir = '/Volumes/research/calps/data3/scratch/adr27/DEPPARSE/treebank3-npbrac-stanforddeps'  
    print count_sentences_in_dir(inputdir,filter="wsj")
    print count_sentences_with_rel_in_dir(inputdir,"conj",filter="wsj")
        