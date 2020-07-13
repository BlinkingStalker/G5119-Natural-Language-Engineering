#5_1
#question 1.1
import pandas as pd
words=["book","chicken","counter","twig","fast","plant"]

results=[[len(wn.synsets(word,parts_of_speech[key])) for key in parts_of_speech.keys()]for word in words]

df =pd.DataFrame(results,index=words,columns=list(parts_of_speech.keys()))
df

#question 1.2
def distance_to_root(asynset):
    print(asynset.lemma_names())
    hypernyms=asynset.hypernyms()
    if len(hypernyms)==0:
        #reached the top and have to stop
        return 0
    else:
        if len(hypernyms)>1:
            print("Warning: multiple hypernyms")
        return (distance_to_root(hypernyms[0])+1)
    
#question 2.1
def path_similarity(wordA,wordB,pos=wn.NOUN):
    synsetsA=wn.synsets(wordA,pos)
    synsetsB=wn.synsets(wordB,pos)
    maxsofar=0
    for synsetA in synsetsA:
        for synsetB in synsetsB:
            sim=wn.path_similarity(synsetA,synsetB)
            if sim>maxsofar:
                maxsofar=sim
    return maxsofar

#question 2.2
def word_similarity(wordA,wordB,pos=wn.NOUN,measure="path"):
    synsetsA=wn.synsets(wordA,pos)
    synsetsB=wn.synsets(wordB,pos)
    maxsofar=0
    brown_ic=wn_ic.ic("ic-brown.dat")
    for synsetA in synsetsA:
        for synsetB in synsetsB:
            if measure=="path":
                sim=wn.path_similarity(synsetA,synsetB)
            elif measure=="res":
                sim=wn.res_similarity(synsetA,synsetB,brown_ic)
            elif measure=="lin":
                sim=wn.lin_similarity(synsetA,synsetB,brown_ic)
            
            if sim>maxsofar:
                maxsofar=sim
    return maxsofar

#question 3.1
measures=["path","res","lin"]
for measure in measures:
    scores=[]

    for i,triple in enumerate(mcdata):
        scores.append(word_similarity(triple[0],triple[1],measure=measure))
    df[measure]=scores
    
df

#5.2
#question 1.1
def simplifiedLesk(word,sentence):
    '''
    Use the simplified Lesk algorithm to disambiguate word in the context of sentence
    word: a String which is the word to be disambiguated
    sentence: a String which is the sentence containing the word
    :return: a pair (chosen sense definition, overlap score)
    '''
    
    #construct the set of context word tokens for the sentence: all words in sentence - word itself
    lemma =WordNetLemmatizer()
    contexttokens=set((filter_stopwords(normalise(word_tokenize(sentence)))))-{word}
    contextlemmas={lemma.lemmatize(contexttoken) for contexttoken in contexttokens}
    
    #get all the possible synsets for the word
    synsets=wn.synsets(word)
    scores=[]
    
    #iterate over synsets
    for synset in synsets:
        #get the set of tokens in the definition of the synset
        sensetokens=set(filter_stopwords(normalise(word_tokenize(synset.definition()))))
        senselemmas={lemma.lemmatize(token) for token in sensetokens}
        #find the size of the intersection of the sensetokens set with the contexttokens set
        scores.append((synset.definition(),len(senselemmas.intersection(contextlemmas))))
    
    #sort the score list in descending order by the score (which is item with index 1 in the pair)
    sortedscores=sorted(scores,key=operator.itemgetter(1),reverse=True) 
    #print(sortedscores)
    return sortedscores[0]
    
#question 2.1
def adaptedLesk(word,sentence):
    '''
    Use the simplified Lesk algorithm to disambiguate word in the context of sentence, using standard WordNet adaptations
    word: a String which is the word to be disambiguated
    sentence: a String which is the sentence containing the word
    :return: a pair (chosen sense definition, overlap score)
    '''
    
    #construct the set of context word tokens for the sentence: all words in sentence - word itself
    
    lemma =WordNetLemmatizer()
    contexttokens=set((filter_stopwords(normalise(word_tokenize(sentence)))))-{word}
    contextlemmas={lemma.lemmatize(contexttoken) for contexttoken in contexttokens}
    #get all the possible synsets for the word
    synsets=wn.synsets(word)
    scores=[]
    
    #iterate over synsets
    for synset in synsets:
        #get the set of tokens in the definition of the synset
        sensetokens=word_tokenize(synset.definition())
        sensetokens+=synset.lemma_names()
        for hypernym in synset.hypernyms():
            sensetokens+=hypernym.lemma_names()
            sensetokens+=word_tokenize(hypernym.definition())
        for hyponym in synset.hyponyms():
            sensetokens+=hyponym.lemma_names()
            sensetokens+=word_tokenize(hyponym.definition())
        
        sensetokens=set(filter_stopwords(normalise(sensetokens)))
        senselemmas={lemma.lemmatize(token) for token in sensetokens}
        #find the size of the intersection of the sensetokens set with the contexttokens set
        scores.append((synset.definition(),len(senselemmas.intersection(contextlemmas))))
    
    #sort the score list in descending order by the score (which is item with index 1 in the pair)
    sortedscores=sorted(scores,key=operator.itemgetter(1),reverse=True) 
    #print(sortedscores)
    return sortedscores[0]
    
#question 3.1
def max_sim(word,contextlemmas,pos=wn.NOUN):
    
    synsets=wn.synsets(word,pos)
    scores=[]
    for synset in synsets:
        total=0
        for lemma in contextlemmas:
            sofar=0
            for synsetB in wn.synsets(lemma,pos):
                sim=wn.path_similarity(synset,synsetB)
                if sim>sofar:
                    sofar=sim
            total+=sofar
        scores.append((synset.definition(),total))
    sortedscores=sorted(scores,key=operator.itemgetter(1),reverse=True) 
    #print(sortedscores)
    return sortedscores[0]

