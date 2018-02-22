#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:53:34 2018

@author: omar
"""


docA = open("Dic.sgm", "r")
content= docA.read()
docB = open("Dic1.sgm", "r")
content1= docB.read()

bowA = content.split(" ")
bowB = content1.split(" ")

wordSet = set(bowA).union(set(bowB))

wordDictA = dict.fromkeys(wordSet, 0)
wordDictB = dict.fromkeys(wordSet, 0)

for word in bowA:
    wordDictA[word]+=1
for word in bowB:
    wordDictB[word]+=1
#lastly I´ll stick those into a matrix
import pandas as pd
pd.DataFrame([wordDictA, wordDictB])

def computeTF(wordDict, bow):
        tfDict = {}
        bowCount = len(bow)
        for word, count in wordDict.iteritem():
            tfDict[word] = count / float(bowCount)
        return tfDict
tfBowA = computeTF(wordDictA, bowA)
tfBowB = computeTF(wordDictB, bowB)

def computeIDF(doclist):
    import math
    idfDict = {}
    N = len(doclist)
    #coun the number of document that contain a word
    idfDict = dict.fromkeys(doclist[0].keys(),0)
    for doc in doclist:
        for word, val in doc.iteritems():
            if val> 0:
                idfDict[word] +=1
                
    #divie N by denominator above, take the log of that
    for word, val in idfDict.iteritems():
        idfDict[word]=math.log(N / float(val))
    return idfDict
idfs = computeIDF([wordDictA, wordDictB])

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.iteritems():
        tfidf[word] = val * idfs[word]
    return tfidf
tfidfBowA = computeTFIDF(tfBowA, idfs)
tfidfBowB = computeTFIDF(tfBowB, idfs)

# lasty Iĺl 
import pandas as pd
pd.DataFrame ([tfidfBowA, tfidfBowB])

    


print (tfidfBowA)

