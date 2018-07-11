# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 13:29:42 2016

@author: victor
"""

import numpy as np
import pandas as pd
from decimal import Decimal
#import sompy
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import log_loss
import matplotlib.pylab as plt
from scipy import sparse
from sklearn.model_selection import StratifiedKFold


class Evaluation(object):
     
     
    def __init__(self,p,pop):
        
        self.p = p
        self.nfolds=10
        self.calls=0
        self.evolucao = []
        self.best = 0.0
        self.pop = pop
        self.atributos = self.p.getNumAtributos() - 1
        self.totaliter = []
        self.data = self.p.getInstances()
        self.label = self.p.label
        
        #self.indices_treino = self.p.indices_treino
        #self.indices_validacao = self.p.indices_validacao

        #self.xgb = self.p.xgb
        
        
    def avalia(self,populacao):
        self.populacao = populacao
        tam = self.populacao.shape[1]-1
                                 
        print("\n")
        #X_train, X_test, y_train, y_test = train_test_split(self.data, self.label, test_size=0.30, random_state=7)
        X_train = self.data
        y_train = self.label
        
        #indices_treino = self.indices_treino
        #indices_validacao = self.indices_validacao
        kfold = 5
        kf = StratifiedKFold(n_splits=kfold, random_state=42)
       
    
        for i in range(0,self.pop):
            print("particula",i)
            lista = self.excluir_coluna(self.populacao[i,:])
            
            '''print("Qtd de features analisadas", self.p.numAtributos-len(lista))'''
            if (self.p.numAtributos - len(lista)) > 1:
                '''print("entrei")'''
                if isinstance(X_train, pd.DataFrame):
        
                   dados_train = X_train.drop(X_train.columns[[lista]],1)
                   #dados_test = X_test.drop(X_test.columns[[lista]],1)
                   '''self.dados = np.asarray(self.dados)'''
                
                if sparse.issparse(X_train):
                   dados_train = X_train[:,lista]
                   #dados_test = X_test[:,lista]
                    
                pred = model1(np.array(dados_train),y_train,kf)
                print(pred)
                #pred = round(log_loss(y_test,agl.predict_proba(dados_test)),3)
                self.populacao[i,tam] = pred
                self.totaliter.append(self.populacao[i,:])              
                 
                   
                
        return self.populacao
        
    

    
    def excluir_coluna(self,s):
           lista = []
           
           for i in range(0,len(s)-1):
               
               if s[i] <= 0.6:
                  lista.append(i)
                  
      
           return lista
     
   
    
   




                  
      
    

