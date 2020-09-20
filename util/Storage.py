# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:29:51 2019

@author: norok
"""
import pickle
import os

class spamStorage:
    def __init__(self, rootdir):
        self.outpath = rootdir + 'Result/'
        if not os.path.exists(rootdir + 'Result/'):
            os.mkdir(rootdir + 'Result/')
        
    def save(self, data, filename):
        filepath = self.outpath + filename
        with open(filepath, mode='wb') as f:
            pickle.dump(data, f)

    def load(self, filename):
        filepath = self.outpath + filename
        with open(filepath, mode='rb') as f:
            data = pickle.load(f)
        return data