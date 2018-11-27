# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:46:48 2016

@author: rebli
"""


import numpy as np

from sklearn.naive_bayes import GaussianNB
import numpy as np
from decimal import Decimal
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

class SFS_SBS(object):
    
    
    def __init__(self,p):
        self.atributos = p.getNumAtributos() - 1
        self.data = p.getInstances()
        self.y_label = self.data.ix[:,self.atributos]
        self.data = self.data.drop(self.data.columns[self.atributos],1)
    
    def SFS(self):
           knn = KNeighborsClassifier(n_neighbors=3)
           self.sfs1 = SFS(knn, 
                      k_features=(1,self.atributos-1), 
                      forward=True, 
                      floating=True, 
                      scoring='accuracy',
                      cv=10)
           
           data = self.data.as_matrix()
           self.sfs1.fit(data, self.y_label)
           self.df = pd.DataFrame.from_dict(self.sfs1.get_metric_dict()).T
           self.df.sort_values('avg_score', inplace=True, ascending=False)
           print('best combination (ACC: %.3f): %s\n' % (self.df[0:1]['avg_score'], 
                 self.df[0:1]['feature_idx']))
           
           
    def SBS(self):
           knn = KNeighborsClassifier(n_neighbors=3)
           self.sfs5 = SFS(knn, 
                      k_features=(1,self.atributos-1), 
                      forward=False, 
                      floating=True, 
                      scoring='accuracy',
                      cv=10)
           
           data = self.data.as_matrix()
           self.sfs5.fit(data, self.y_label)
           self.df_2 = pd.DataFrame.from_dict(self.sfs5.get_metric_dict()).T
           self.df_2.sort_values('avg_score', inplace=True, ascending=False)
           print('best combination (ACC: %.3f): %s\n' % (self.df_2[0:1]['avg_score'], 
                 self.df_2[0:1]['feature_idx']))
           
           
           
           
           
           
           