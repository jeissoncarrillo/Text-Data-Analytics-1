# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 08:42:12 2018

@author: MILO
"""
import pymongo

from pymongo import MongoClient

def conexionBD():
    #Conx a mongo
    client = MongoClient()
    client =("")
    db = client.prueba1
    return db
  
  
from __future__ import print_function
from time import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import array

db = conexionBD()
#tablas de contenido
Noticia    = db.noticias
Raw        = db.raw
StopWord   = db.stop
MatrizTfTd = db.tfitd


n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20

t0 = time()
print("Loading dataset and extracting TF-IDF features...\n\n")


#Eliminar información no requerida
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))

 #Navegador de preguntas 
print("\n bloques de texto para clasificar:%s"% len(dataset.data))

#Contenido sin etiquecas Html o xml
print(dataset.data[0])


#Eliminar Stopwords
words = dataset.data[0].lower().split()


#Eliminar los StopWords
addictional_stopwords = " <nfsi> , . ' ¿ '' + } { - = : )( ¬ ] [ ^ ? </ < \> > & # $ % ! ¡ 's 1 2 3 4 5 6 7 8 9 0 25 19.6 a bout above about all, got across after afterwards again against all almost alone along already also although always am among amongst amoungst amount an and another any anyhow anyone anything anyway any where are around as at back be became because become becomes becoming been before beforehand behind being below beside besides between beyond bill both bottom but by call can cannot cant co computer con could couldnt cry de describe detail do done down due during each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except few fifteen fify fill find fire first five for former formerly forty found four from front full further get give go had has hasnt have he hence her here hereafter hereby herein hereupon hers herse him himse his how however hundred i ie if in inc indeed interest into is it its itse keep last latter latterly least less ltd made many may me meanwhile might mill mine more moreover most mostly move much must my myse name namely neither never nevertheless next nine no nobody none noone nor not nothing now nowhere of off often on once one only onto or other others otherwise our ours ourselves out over own part per perhaps please put rather re same see seem seemed seeming seems serious several she should show side since sincere six sixty so some somehow someone something sometime sometimes somewhere still such system  take ten thanthat the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin third this those though three through throughout thru thus to together too top toward towards twelve twenty two un under until up upon us very via was  we well were what whatever when whence whenever where whereafter whereas whereby  wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would yet you your yours yourself yourselves I U.S i'm that  u.s u.s."
dividirsw = addictional_stopwords.split()
wordsFiltered = [w for w in words if not w in dividirsw]
print("Diccionario:")
print (json.dumps(wordsFiltered))

#guardar sin stopwords en un txt
limpio = open('SinStop.sgm','wt')
limpio.write('\n'.join(wordsFiltered))
limpio.close()

#eliminar palabras repetidas crear diccionario
dic_limpio = []
for r in wordsFiltered:
    if not r in dic_limpio:
        dic_limpio.append(r)
d1=open ('Dic.sgm','wt')
d1.write('\n'.join(dic_limpio))
d1.close()




#Bolsa de palabras

vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                             stop_words='english')
#tfidf = vectorizer.fit_transform(dataset.data[:n_samples])
tfidf = TfidfVectorizer().fit_transform(dataset.data)
print("\n done in %0.3fs." % (time() - t0))
#Vector matriz
#print(tfidf[0:1])


#coseno de similitud

from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
#print (cosine_similarities)

related_docs_indices = cosine_similarities.argsort()[:-6:-1]
#print (related_docs_indices)
cosine_similarities[related_docs_indices]

#print (dataset.data[6057])



#buscador
"""print("\n Digite las palabras claves")
palabra=input()

print("\n Su palabra fue : %s")


#crear los token
vector=CountVectorizer()
vector.fit(dataset.data)   
#print("\n TOKEN : %s"% vector.vocabulary_)   """
    
