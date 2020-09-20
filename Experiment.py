# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 07:22:22 2019

@author: norok
"""

import os
import json
import sys
sys.path.append("C:/Users/norok/Documents/Python Scripts/signate/spam/util")

from Dataset import Dataset_spam
from Transformer import textCleanser, transformBoW
from Trainer import trainGS
from Storage import spamStorage
def main():    
    rootdir = "C:/Users/norok/Documents/Python Scripts/signate/spam/"
    os.chdir(rootdir)
    
    if not os.path.exists(rootdir + 'config/'):
        os.mkdir(rootdir + 'config/')
    
    jsonfile = rootdir + 'config/param.json'
    with open(jsonfile, mode='rb') as f:
        params = json.load(f)
    
    for i in range(0,len(params['params'])):
        param = params['params'][i]
        print(i)
        pipeline(rootdir, param)

def pipeline(rootdir, params):
    dataset_spam = Dataset_spam(rootdir)
    dataset_spam.load_spam()
    dataset_spam.describe()

    cleanse_docs = textCleanser(isStopword = params['isStopword'])
    dt_train = cleanse_docs.transform(dataset_spam.data)
    print(len(dt_train))
    print(params['isTFIDF'])
    bowtransformer = transformBoW(isTFIDF = params['isTFIDF'])
    bowtransformer.fit(dt_train)
    X = bowtransformer.transform(dt_train)
    y = dataset_spam.label
    print(X.shape, y.shape, params['identifier'])
    traings = trainGS(params['trainParams'])
    bestmodel = traings.train(X, y, params['identifier'])

    spamstorage = spamStorage(rootdir)
    spamstorage.save(cleanse_docs, params['docsCleanser'])
    spamstorage.save(bowtransformer, params['bowTransformer'])
    spamstorage.save(bestmodel, params['bestmodelFile'])

if __name__ == '__main__':
    main()