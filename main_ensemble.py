# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 23:49:42 2017

@author: rebli
"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.cross_validation import KFold;
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report
from sklearn import metrics
import pickle
import sys
sys.path.insert(0, 'features/fenginner.py')
import fenginner as enginner
sys.path.insert(0, 'pso/Principal.py')
import Principal as principal



class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None, flag=0):
        print(clf)
        if flag == 1:
           #print('amigo') 
           self.clf = clf.XGBClassifier(**params)
        else:   
          params['random_state'] = seed
          self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)
    
    def predict(self, x):
        return self.clf.predict(x)
    
    
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_
    
def get_oof(clf, x_train, y_train):
    oof_train = np.zeros((ntrain,2))
   
    
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)
        
        
           
        oof_train[test_index,:] = clf.predict_proba(x_te)
        
    return oof_train

def get_oof_all_data(clf, x_train, y_train):
    
    clf.train(x_train, y_train)
    



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    


def model_xgboost(xgb_param,X_train,y_train, X_test, y_test, useTrainCV=True, cv_folds=10, early_stopping_rounds=150):
    
    num_boost_rounds = 1000
    if useTrainCV:
        
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_param['n_estimators'], nfold=cv_folds,
            metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        xgb_param['n_estimators'] = cvresult.shape[0]
        num_boost_rounds = len(cvresult)
        print(cvresult.iloc[-1:])
    
    #Fit the algorithm on the data
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(xgb_param, xgtrain, num_boost_round=num_boost_rounds, verbose_eval=10)

    
    xgtest = xgb.DMatrix(X_test, label=y_test) 
    #Print model report:
    pred_test_y = model.predict(xgtest)
    pred_train = model.predict(xgtrain)
    if xgb_param['objective'] == 'multi:softprob':
      print "\nModel Report"
      print "Accuracy Treino: %.4g" % log_loss(y_train,pred_train)
      
    if xgb_param['objective'] == 'multi:softmax':
      print "\nModel Report"
      print "Accuracy Treino: %.4g" % accuracy_score(y_train,pred_train)
      
      
    '''feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')'''
    
    return pred_test_y, model


def model_final_xgboost(xgb_param,X_train,y_train, useTrainCV=True, cv_folds=10, early_stopping_rounds=150):
    
    num_boost_rounds = 1000
    if useTrainCV:
        
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_param['n_estimators'], nfold=cv_folds,
            metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        xgb_param['n_estimators'] = cvresult.shape[0]
        num_boost_rounds = len(cvresult)
        print(cvresult.iloc[-1:])
    
    #Fit the algorithm on the data
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(xgb_param, xgtrain, num_boost_round=num_boost_rounds, verbose_eval=10)

    
    #xgtest = xgb.DMatrix(X_test, label=y_test) 
    #Print model report:
    #pred_test_y = model.predict(xgtest)
    pred_train = model.predict(xgtrain)
    if xgb_param['objective'] == 'multi:softprob':
      print "\nModel Report"
      print "Accuracy Treino: %.4g" % log_loss(y_train,pred_train)
      
    if xgb_param['objective'] == 'multi:softmax':
      print "\nModel Report"
      print "Accuracy Treino: %.4g" % accuracy_score(y_train,pred_train)
      
      
    '''feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')'''
    
    return model

    
                    




if __name__ == '__main__':
   
   print('Lendo a base de dados.....') 
   data = pd.read_csv('data/dataset.csv')
   description = pd.read_csv('data/description.csv', header=None)        
   cols = description[0].values
   data = data[cols]
   
   #Separando os dados para treino e teste para avaliar o modelo de baseline apenas com variáveis númericas
   
   data,label = enginner.filtrar_clientes(data)
   data = enginner.exclui_variavel(data,['id','loan_term'])
   print('Iniciando o feature Enginner')
   data,count_values_zip_code,subs_mariral_status,subs_tudo,lista_city,vectorizer,contagem_landingpage,columnsToEncode,dicionario = enginner.feature_enginner(data)
   #data = enginner.standarlizar_dados(data)
  
   ## Rodando o segundo modelo     
   #X_treino, X_teste, label_treinamento, label_teste = train_test_split(data,label,test_size=0.30, random_state=42)
   
   
   
   best_global = np.loadtxt('best_global.txt')
   lista = enginner.excluir_coluna(best_global)
   
   #melhor  = principal.iniciar(data,label) 
   #lista = enginner.excluir_coluna(best_global)

   
   data = data[lista]   
   data.replace([np.inf, -np.inf], -10, inplace=True) 
   data.fillna(-10, inplace=True)  


   print('Iniciando o treinamento do ensemble')          
   ntrain = data.shape[0]
   SEED = 0 
   NFOLDS = 5 
   kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

   rf_params_1 = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : None,
    'verbose': 0
   }

   rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
    }

   # AdaBoost parameters
   ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
   }

  # Gradient Boosting parameters
   gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
  }


  # Extra Trees Parameters
   et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
   }

   xgb_params = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'num_class': 2,
    'subsample': 0.85,
    'n_estimators':1000,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'silent': 1
                 }  


   rf_1 = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params_1)
   rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
   et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
   ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
   gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
   xgbo = SklearnHelper(clf=xgb, seed=SEED, params=xgb_params, flag=1)


   y_train = label.values
   x_train = data.values 
   print('Iniciando o treinamento dos classificadores de primeiro nivel')
   print('Iniciando o ExtraTree')
   et_oof_train = get_oof(et, x_train, y_train) # Extra Trees
   
   print('Iniciando o RandomForest_1')
   rf_oof_train= get_oof(rf,x_train, y_train) # Random Forest
   
                        
   print('Iniciando o RandomForest_2')
   rf_oof_train_1= get_oof(rf_1,x_train, y_train)
   
   print('Iniciando o Adaboost')
   ada_oof_train = get_oof(ada, x_train, y_train) # AdaBoost 
    
   print('Iniciando o Gradient Boost')
   gb_oof_train = get_oof(gb,x_train, y_train) # Gradient Boost
    
   print('Iniciando o Extreme Gradient Boost')
   xgbo_oof_train = get_oof(xgbo,x_train, y_train) # Extreme Gradient Boost

   x_train = np.concatenate((rf_oof_train, ada_oof_train, gb_oof_train,xgbo_oof_train,rf_oof_train_1,et_oof_train), axis=1)

   X_treino, X_teste, label_treina, label_tes = train_test_split(x_train,y_train,test_size=0.30, random_state=42)

   xgb_params1 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'num_class': 2,
    'subsample': 0.85,
    'n_estimators':1000,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'silent': 1
                 }   
   
   print('Iniciando o treinamento do classificador de segundo nivel')
   pred_test_y, model1 = model_xgboost(xgb_params1,X_treino,label_treina, X_teste, label_tes, useTrainCV=True, cv_folds=5, early_stopping_rounds=100)
   print('Acuracia final obtida:')
   print(accuracy_score(label_tes, pred_test_y))
   print('Matriz de confusao do modelo final:')
   print(confusion_matrix(label_tes, pred_test_y))
   
   
   target_names = ['nao-enviado', 'enviado']
   print('Conjunto de métricas')
   print(classification_report(label_tes, pred_test_y, target_names=target_names))
   
   xgb_params1 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'num_class': 2,
    'subsample': 0.85,
    'n_estimators':1000,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'silent': 1
                 } 
   
   pred_test_y, model1 = model_xgboost(xgb_params1,X_treino,label_treina, X_teste, label_tes, useTrainCV=True, cv_folds=5, early_stopping_rounds=100)
   
   
   
   prob = []

   for i in range(0,len(label_tes)):
       prob.append(pred_test_y[i,label_tes[i]])

   fpr, tpr, thresholds = metrics.roc_curve(label_tes, prob, pos_label=0)
   print('Metrica AUC')
   print(metrics.auc(fpr, tpr))


###############################################
# Retreinando tudo de novo com todos os dados para salvar modelo.
   

   xgb_params1 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'num_class': 2,
    'subsample': 0.85,
    'n_estimators':1000,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'silent': 1
                 }   
   
   print('Iniciando o treinamento do classificador de segundo nivel')
   model1 = model_final_xgboost(xgb_params1,x_train, y_train,useTrainCV=True, cv_folds=5, early_stopping_rounds=100)
   
   
   
   
   rf_1 = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params_1)
   rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
   et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
   ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
   gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
   xgbo = SklearnHelper(clf=xgb, seed=SEED, params=xgb_params, flag=1)
   
   
   y_train = label.values
   x_train = data.values 
   print('Iniciando o treinamento dos classificadores de primeiro nivel')
   print('Iniciando o ExtraTree')
   get_oof_all_data(et, x_train, y_train) # Extra Trees
   
   print('Iniciando o RandomForest_1')
   get_oof_all_data(rf,x_train, y_train) # Random Forest
   
                        
   print('Iniciando o RandomForest_2')
   get_oof_all_data(rf_1,x_train, y_train)
   
   print('Iniciando o Adaboost')
   get_oof_all_data(ada, x_train, y_train) # AdaBoost 
    
   print('Iniciando o Gradient Boost')
   get_oof_all_data(gb,x_train, y_train) # Gradient Boost
    
   print('Iniciando o Extreme Gradient Boost')
   get_oof_all_data(xgbo,x_train, y_train) # Extreme Gradient Boost

   
   




   pickle.dump(count_values_zip_code,open("objetos/count_values_zip_code.p", "wb"))
   pickle.dump(subs_mariral_status,open("objetos/subs_mariral_status.p", "wb"))
   pickle.dump(subs_tudo,open("objetos/subs_tudo.p", "wb"))  
   pickle.dump(lista_city,open("objetos/lista_city.p", "wb"))
   pickle.dump(vectorizer,open("objetos/vectorizer.p", "wb"))  
   pickle.dump(contagem_landingpage,open("objetos/contagem_landingpage.p", "wb"))  
   pickle.dump(columnsToEncode,open("objetos/columnsToEncode.p", "wb"))  
   pickle.dump(dicionario,open("objetos/dicionario.p", "wb"))  
   pickle.dump(model1,open("objetos/model_final.p", "wb"))

   pickle.dump(rf_1,open("objetos/randomforest_2.p", "wb"))
   pickle.dump(rf,open("objetos/randomforest_1.p", "wb"))
   pickle.dump(et,open("objetos/extratree.p", "wb"))
   pickle.dump(ada,open("objetos/adaboost.p", "wb"))
   pickle.dump(gb,open("objetos/gradientboostin.p", "wb"))
   pickle.dump(xgbo,open("objetos/extremegradientboosting.p", "wb"))







'''from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target


import numpy as np
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
metrics.calinski_harabaz_score(X, labels)  


from sklearn.cluster import AffinityPropagation
af = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)'''



















