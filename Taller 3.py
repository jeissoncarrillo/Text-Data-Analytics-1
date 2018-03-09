

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:02:13 2018

@author: Jeisson Carrillo
@author: Omar Ocampo
"""

import time
import re

from bs4 import BeautifulSoup   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


doc = open("reut2-000.sgm","r")
soup = BeautifulSoup(doc, 'html,parser')
#creando arreglos y variables 
doc.close()
noticias=[]
logn=[]

noticas= soup.find.all('reutera')
#quitar caracteres especials
#dejar datos listos para la lista
fileRaw = open("raw.sgm","w")
logn=[]
for i in range(len(noticias)):
    try:
        cadena = noticias[i].title.string.replace('\n',' ')+" "+noticias[i].body.string.replace('\n',' ')
        
        cadena = noticias[i].title.string +"@@,"+ noticias[i].body.string + "@@;"
        cadena = cadena.lower()
        cadena = re.sub(r'<.*>|[0-9]|[,*$]|[.*$]|[-*$]|[(.*)$]|[/*$]|["*$]|[\'][a-z|\W]|[+*$]|[:*$]'," ",cadena) 
        cadena = cadena.replace ('reuter','')
        cadena = cadena.replace('\n', '')
        cadena = cadena.replace('@@;','' )
        cadena = cadena.replace('@@','' )
        cadena = cadena.replace('\x03','' )
        logn.append(cadena)
        fileRaw.write(cadena+"\n")
    except:
        pass
    
#Calculo de longitud
lenv=len(logn)       
n_features = 1000
n_topics = 10
n_top_words = 20

#Tiempo para iniciar
t0 = time()
print("Loading dataset and extracting TF-IDF features...")

vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                             stop_words='english')
tfidf = vectorizer.fit_transform(noticias[lenv])
doc2=open("tfidf.sgm","w")
doc2.write(str(tfidf))
fileRaw.close()
doc2.close()

print(tfidf)

# Fit the NMF model
print("Fitting the NMF model with n_samples=%d and n_features=%d..."
      % (nlen, n_features))
nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

feature_names = vectorizer.get_feature_names()


for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
