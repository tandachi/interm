# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:01:04 2017

@author: rebli
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 23:56:25 2017

@author: rebli
"""

'''mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev0\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']'''

import os
import numpy as np
import xgboost as xgb
import  lightgbm as lgb
import itertools
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

#metrics = {'auc': roc_auc_score}

def ctun_model_manual(data,y_train,estimator,kf, eval_metric='auc' ,early_stopping_rounds = 100, **kargs):
    
    print("Iniciando o tunning.....")
    type_estimator = estimator.__module__
    
    if 'xgboost' in type_estimator:
        print(kargs)
        #dat = xgb.DMatrix(data,y_train)
        cvs = {}
        gb =  [dict(zip(kargs.keys(), p)) for p in [x for x in apply(itertools.product, kargs.values())]]
        print('Serão testados:{}'.format(len(gb)))
        for i in gb:
            print("Modelo",i)
            folds = []
            estimator.set_params(**i)
            for p, (train_index, test_index) in enumerate(kf.split(data, y_train)):
                    train_X, valid_X = data.iloc[train_index], data.iloc[test_index]
                    train_y, valid_y = y_train[train_index], y_train[test_index]
                    #print('teste')
                    estimator.fit(train_X, train_y,eval_set=[(valid_X,valid_y)], eval_metric=eval_metric,
                                  early_stopping_rounds=early_stopping_rounds, verbose=False)
                    #print('teste1')
                    folds.append(estimator.best_score)
            #print('dd',np.mean(folds))        
            cvs[frozenset(i.items())] = np.mean(folds)
            
    elif 'lightgbm' in type_estimator:
        print(kargs)
        #dat = xgb.DMatrix(data,y_train)
        cvs = {}
        gb =  [dict(zip(kargs.keys(), p)) for p in [x for x in apply(itertools.product, kargs.values())]]
        print('Serão testados:{}'.format(len(gb)))
        for i in gb:
            print("Modelo",i)
            folds = []
            estimator.set_params(**i)
            for p, (train_index, test_index) in enumerate(kf.split(data, y_train)):
                    train_X, valid_X = data.iloc[train_index], data.iloc[test_index]
                    train_y, valid_y = y_train[train_index], y_train[test_index]
                    #print('teste')
                    estimator.fit(train_X, train_y,eval_set=[(valid_X,valid_y)], eval_metric=eval_metric,
                                  early_stopping_rounds=early_stopping_rounds, verbose=False)
                    #print('teste1')
                    folds.append(estimator.evals_result['valid_0']['auc'][-1])
            #print('dd',np.mean(folds))        
            cvs[frozenset(i.items())] = np.mean(folds)
    else:
        #print(kargs)
        cvs = {}
        gb =  [dict(zip(kargs.keys(), p)) for p in [x for x in apply(itertools.product, kargs.values())]]
        print('Serão testados:{}'.format(len(gb)))
        for i in gb:
            print("Modelo",i)
            folds = []
            estimator.set_params(**i)
            for p, (train_index, test_index) in enumerate(kf.split(data, y_train)):
                    train_X, valid_X = data.iloc[train_index], data.iloc[test_index]
                    train_y, valid_y = y_train[train_index], y_train[test_index]
                    
                    estimator.fit(train_X, train_y)
                    
                    if eval_metric == 'auc':
                       pred = estimator.predict_proba(valid_X)[:,1]
                    
                       folds.append( roc_auc_score(valid_y, pred))
            #print('dd',np.mean(folds))        
            cvs[frozenset(i.items())] = np.mean(folds)
            
            
    return cvs


     
