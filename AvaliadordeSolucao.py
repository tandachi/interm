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


class AvaliadordeSolucao(object):
     
     
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
     
   
    
   
def model1(data,y_train,kf):
    
    #num_vali = int(len(data) * 0.20)
    
    
    #x_train = data.iloc[indices_treino]
    #y_train1 = y_train[indices_treino]
    
    #x_valid = data.iloc[indices_validacao]
    #y_valid = y_train[indices_validacao]
    


   xgb_params_2 = {
    'eta': 0.02,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'gamma' :10,
    'reg_alpha': 8,
    'tree_method': 'hist',
    'silent': True
     
                 }
   lista_models = []
   for i, (train_index, test_index) in enumerate(kf.split(data, y_train)):
        train_X, valid_X = data[train_index], data[test_index]
        train_y, valid_y = y_train[train_index], y_train[test_index]
    
        d_train = xgb.DMatrix(train_X, train_y)
        d_valid = xgb.DMatrix(valid_X, valid_y)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        model = xgb.train(xgb_params_2, d_train, 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=False, early_stopping_rounds=100)
        lista_models.append(model.best_score)

    


    
   return np.mean(lista_models)

    
        
           
def modelfit(alg, X_train,y_train, useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    
    
    '''xgb_param = alg.get_xgb_params()
    xgb_param['num_class'] = 3
    if useTrainCV:'''
        
        #xgtrain = xgb.DMatrix(X_train, label=y_train)
    
    '''cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])'''
           
    #Fit the algorithm on the data
    
    alg.fit(X_train, y_train,eval_metric='mlogloss')
        
    #Predict training set:
    
   # dtrain_predprob = alg.predict_proba(X_train)
        
    #Print model report:
    
    '''print "\nModel Report"
    print "Accuracy Treino: %.4g" % log_loss(y_train, dtrain_predprob)'''
    
                    
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
    return alg
                
def gini(actual, pred, cmpcol = 0, sortcol = 1):
        assert( len(actual) == len(pred) )
        all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
        all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
        totalLosses = all[:,0].sum()
        giniSum = all[:,0].cumsum().sum() / totalLosses
    
        giniSum -= (len(actual) + 1) / 2.
        return giniSum / len(actual)
 
def gini_normalized(a, p):
       return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
       labels = dtrain.get_label()
       gini_score = gini_normalized(labels, preds)
       return 'gini', gini_score           



                  
      
    

