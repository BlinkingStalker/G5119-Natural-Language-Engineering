#Selected solutions for week 7
#Part 1
#Ex 1.1
def find_sense_distributions(some_sentences):
    allwords={}
    for sentence in some_sentences:
        for (word,sense) in sentence:
            thisword=allwords.get(word,{})
            thisword[sense]=thisword.get(sense,0)+1
            allwords[word]=thisword
    return allwords
#Ex 1.2
def find_monosemous(sense_dists):
    mono=[]
    for key,worddict in sense_dists.items():
        if len(worddict.keys())==1:
            mono.append((key,sum(worddict.values())))
    return sorted(mono,key=operator.itemgetter(1),reverse=True)
#Ex 1.3
def find_candidates(sense_dists):
    cands=[]
    for key,worddict in sense_dists.items():
        if len(worddict.keys())==2:
            freq=sum(worddict.values())
            p=list(worddict.values())[0]/freq
            if p>0.3 and p<0.7:
                cands.append((key,freq,p))
    return sorted(cands,key=operator.itemgetter(1),reverse=True)

#Ex 2.1
def evaluate(cls,test_data):
    correct=0
    wrong=0
    predictions={}
    actual={}
    for doc,label in test_data:
        prediction=cls.classify(doc)
        predictions[prediction]=predictions.get(prediction,0)+1
        actual[label]=actual.get(label,0)+1
        if prediction==label:
            correct+=1
        else:
            wrong+=1
    acc=correct/(correct+wrong)
    print("Accuracy of NB classification on testing data is {} out of {}".format(correct,correct+wrong))
    

#Ex 2.3
def train_and_test(word):
    training=get_training_data(training_sentences,word)
    testing=get_training_data(testing_sentences,word)
    classifier=NaiveBayesClassifier.train(training)
    #evaluate(classifier,testing)
    evaluate_precision(classifier,testing)
    
##Part 2
#Ex 1.1
def find_words(word,relation):
    synsets=wn.synsets(word)
    words=set()
    if relation.startswith("syn"):
        words.update(set(flatten([[lemma.name() for lemma in synset.lemmas()] for synset in synsets])))
    elif relation.startswith("ant"):
        words.update(set(flatten([[[antonym.name() for antonym in lemma.antonyms()] for lemma in synset.lemmas()] for synset in synsets])))
        
    elif relation.startswith("hypo"):
        words.update(set(flatten([[[lemma.name() for lemma in hyponym.lemmas()] for hyponym in synset.hyponyms()] for synset in synsets])))
    elif relation.startswith("hyper"):
        words.update(set(flatten([[[lemma.name() for lemma in hypernym.lemmas()] for hypernym in synset.hypernyms()] for synset in synsets])))   
    return words

#Ex 1.2
def extend(positive_seeds,negative_seeds,plus_relations,minus_relations):
    pos_plus=list(positive_seeds)
    neg_plus=list(negative_seeds)

    for word in positive_seeds:
        for relation in plus_relations:
            new_words=find_words(word,relation)
            pos_plus+=new_words
        for relation in minus_relations:
            new_words=find_words(word,relation)
            neg_plus+=new_words
    for word in negative_seeds:
        for relation in plus_relations:
            new_words=find_words(word,relation)
            neg_plus+=new_words
        for relation in minus_relations:
            new_words=find_words(word,relation)
            pos_plus+=new_words
    return pos_plus,neg_plus
            
#Ex 1.3
results=[classifier_evaluate(baseline,testing,ms)]
synclass=SimpleClassifier(pos_syn,neg_syn)
results.append(classifier_evaluate(synclass,testing,ms))
#add other classifiers
df=pd.DataFrame(results,columns=["acc","pre","rec","f1"],index=["baseline","syn","ant","hypo","hyper"])
df
df["acc"].plot.bar()

#Ex 2.1
from nltk.corpus import stopwords

stop = stopwords.words('english')

def search_conj(conj,word,corpus):
    candidates=[]
    for i,token in enumerate(corpus):
        if token ==conj:
            if corpus[i-1]==word:
                candidates.append(corpus[i+1])
            elif corpus[i+1]==word:
                candidates.append(corpus[i-1])
    return set([c for c in candidates if c not in stop and c.isalpha()])




    
    
    