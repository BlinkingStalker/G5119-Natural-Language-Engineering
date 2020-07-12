#Exercise 1.1
def dot(docA,docB):
    the_sum=0
    for (key,value) in docA.items():
        the_sum+=value*docB.get(key,0)
    return the_sum

#Exercise 1.2
def cos_sim(docA,docB):
    sim=dot(docA,docB)/(math.sqrt(dot(docA,docA)*dot(docB,docB)))
    return sim

#Exercise 2.1
def idf(doclist):
    N=len(doclist)
    return {feat:math.log(N/v) for feat,v in doc_freq(doclist).items()}

