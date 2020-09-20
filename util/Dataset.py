# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 06:21:34 2019

@author: norok
"""

import glob
import os
import csv
import numpy as np
import collections

def readText(path):
    os.chdir(path)
    files = glob.glob("*")
    docs = []
    sens = ''
    for file in files:
        f = open(file)
        lines = f.readlines()  # ファイル終端まで全て読んだデータを返す
        for i,line in enumerate(lines):
            if i == 0:
                sens = str(line)
            else:
                sens = '{} {}'.format(sens,str(line))                                
        docs.append(sens)
        sens = ''
    f.close()    
    return docs

class Dataset_spam:
    def __init__(self,path):
        self.path_data = path + "train"
        self.path_label = path + "train_master.tsv"
        self.data = None
        self.label = None
        self.data_shape = None
        self.label_shape = None
        self.label_count = None

    def load_spam(self):
    
        tsvFile = open(self.path_label)
        tsv = csv.reader(tsvFile, delimiter = '\t')
        labels = [y[1] for y in tsv]
        self.label = np.array(labels[1:])
        self.data = np.array(readText(self.path_data))

    def batch_iter(self):
        print("Under Construction")
    def describe(self):
        self.data_shape = self.data.shape
        self.label_shape = self.label.shape
        self.label_count = collections.Counter(self.label)

