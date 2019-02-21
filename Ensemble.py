"""
Created on Wed Sep 13 12:01:41 2017

@author: rebli
"""

import numpy as np
from sklearn.metrics import mean_absolute_error    
from sklearn import metrics

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None, flag=0):
        #print(clf)
        self.cvs = []
        self.flag = flag
        if self.flag == 1:
           #print('amigo') 
           self.clf = clf.XGBClassifier(**params)
           #self.clf = clf
        else:   
          params['random_state'] = seed
          self.clf = clf(**params)

    def train(self, x_train, y_train, x_valid, y_valid):
        
        if self.flag == 1:
           eval_set = [(x_valid, y_valid)] 
           self.clf.fit(x_train, y_train, eval_set=eval_set,early_stopping_rounds=100)
        else:
           pass 
    
    def predict_proba(self, x):
        return self.clf.predict_proba(x)
    
    def predict(self, x):
        return self.clf.predict(x)
    
    
    
    def fit(self,x,y,eval_set, early_stopping_rounds=100):
        
        if self.flag == 1:
            return self.clf.fit(x,y,eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, verbose=False)
        else:
            pass
        
        
        
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_
    
    def save_cv(self,y_true,y_pred):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=2)
        
        self.cvs.append(metrics.auc(fpr, tpr))

    def gini(self,actual, pred, cmpcol = 0, sortcol = 1):
          assert( len(actual) == len(pred) )
          all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
          all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
          totalLosses = all[:,0].sum()
          giniSum = all[:,0].cumsum().sum() / totalLosses
    
          giniSum -= (len(actual) + 1) / 2.
          return giniSum / len(actual)
 
    def gini_normalized(self,a, p):
         return self.gini(a, p) / self.gini(a, a)

    def gini_xgb(self,preds, dtrain):
         labels = dtrain.get_label()
         gini_score = self.gini_normalized(labels, preds)
         return 'gini', gini_score
     


