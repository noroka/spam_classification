# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 07:19:03 2019

@author: norok
"""

from sklearn.base import BaseEstimator, TransformerMixin
import re
from nltk.corpus import stopwords
from gensim import corpora, models
import numpy as np

class skPlumberBase(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self


class textCleanser(skPlumberBase):
    def __init__(self, isStopword = True):
        self.isStopword = isStopword

    def transform(self, docs):        
        def cleanseText(doc, isStopword):
            stop_words = stopwords.words('english')            
            doc = doc.lower()
            doc = re.sub(re.compile("[0-9]+"), 'UNK', doc) # 数字を0に変換
            doc = re.sub(re.compile("[!-/:-@[-`{-~]"), ' ', doc) # 記号を消去
            doc = re.sub(re.compile("\\n"), '', doc) # \nを消去
            
            if isStopword: # ストップワードの消去
                for s_word in stop_words:
                    pat = " " + s_word + " "
                    doc = re.sub(pat , ' UNK ', doc)

            doc = re.sub(" {2,}"," ",doc) # スペース2文字以上を1文字のみに変換            
            doc = doc.split(" ")
            return doc
        
        docs_t = [cleanseText(doc, self.isStopword) for doc in docs ]               
        return docs_t 

class transformBoW(skPlumberBase):
    def __init__(self, isTFIDF):
        self.dictionary = None
        self.isTFIDF = isTFIDF

    def fit(self, docs):
        self.dictionary = corpora.Dictionary(docs)
        
    def transform(self, docs):
        
        for i in range(0,len(docs)):
            doc = docs[i]
            for j in range(0,len(doc)):
                word = doc[j]
                if not word in self.dictionary.token2id:
                    docs[i][j] = 'UNK'

        corpus = [self.dictionary.doc2bow(doc) for doc in docs]
        dense = np.zeros([len(docs),len(self.dictionary)])
        
        if self.isTFIDF:
            model_tfidf = models.TfidfModel(corpus)
            tfidf = model_tfidf[corpus]
            for i in range(0, len(tfidf)):
                tfidf_ = tfidf[i]
                for j in range(0, len(tfidf_)):
                    dense[i, tfidf_[j][0]] = tfidf_[j][1]
        else:
            for i in range(0, len(corpus)):
                corpus_ = corpus[i]
                for j in range(0, len(corpus_)):
                    dense[i, corpus_[j][0]] = corpus_[j][1]

        return dense
            
    