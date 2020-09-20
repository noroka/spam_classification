# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 08:11:47 2019

@author: norok
"""

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def selectModel(identifier, isBestParam, bestparam = None):    
    if isBestParam:
        if identifier == "MNB":
            model = MultinomialNB(**bestparam)
        elif identifier == "RF":
            model = RandomForestClassifier(**bestparam)
        elif identifier == "SVM":
            model = SVC(**bestparam)
        else:
            print("identifier should be MNB/RF/SVM")
    else:
        if identifier == "MNB":
            model = MultinomialNB()
        elif identifier == "RF":
            model = RandomForestClassifier()
        elif identifier == "SVM":
            model = SVC()
        else:
            print("identifier should be MNB/RF/SVM")
               
    return model

class trainGS:
    def __init__(self,params, cv = 5):
        self.params = params
        self.cv = cv
        self.best_score = None
        self.best_param = None

    def train(self, X, y, identifier):
        clf = selectModel(identifier, False)
        grid_search = GridSearchCV(clf,  # 分類器を渡す
                                   param_grid=self.params,  # 試行してほしいパラメータを渡す
                                   cv=self.cv,  # k-Fold CV で汎化性能を調べる
                                   )
        grid_search.fit(X, y)
        self.best_score = grid_search.best_score_
        self.best_param = grid_search.best_params_
        
        clf_bp = selectModel(identifier, True, self.best_param)
        clf_bp.fit(X, y)
        
        return clf_bp