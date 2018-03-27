# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:46:39 2017

@author: rebli
"""



import numpy as np
import scipy.io
import unicodedata
from sklearn.model_selection import train_test_split
#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import itertools
import sys
sys.path.insert(1, 'CWRU/')
#sys.path.insert(0, 'CWRU/classifyMLP.py')
#sys.path.insert(0, 'Ensemble.py')
#sys.path.insert(0, 'funcoes.py')
sys.path.insert(1, 'pso/')
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

import Principal as principal
reload(principal)
import classifyConvNet as classifyConvNet
#import classifyMLP as classifyMLP
import Ensemble as Ensemble
reload(Ensemble)
import funcoes as funcoes
reload(funcoes)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn import preprocessing

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from keras.utils import np_utils            

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#from numpy.random import seed
#seed(1333)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
#from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical 
from keras.layers import Embedding

from matplotlib.font_manager import FontProperties


####################################################################    

dados_crus = funcoes.get_data('dados/dados_crus.mat','M')
label = dados_crus[:,-1]
dados_crus = np.delete(dados_crus, -1, axis=1)

X_treino, X_teste, label_treina, label_tes = train_test_split(dados_crus,label,test_size=0.20, random_state=42)

X_treino = preprocessing.scale(X_treino)
X_teste = preprocessing.scale(X_teste)

X_treino = X_treino.reshape(X_treino.shape + (1,))
X_teste = X_teste.reshape(X_teste.shape + (1,))


'''model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(55, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])  

model.fit(X_treino, np_utils.to_categorical(label_treina), batch_size=32, epochs=15, verbose=1)

              
fv = model.predict_classes(X_teste)
fv = fv.reshape((1,-1))[0]
classe = ['normal', 'falha']
funcoes.plot_confusion_matrix(confusion_matrix(label_tes,fv),classe)'''

##########################################################################
#input_shape = (d, chan)
kernel_size = 4# size of the convolution filter (image would be e.g. tupel (3,3) )
filters = 26 #     
          
model = Sequential()              
convlay1 = Conv1D(input_shape=(5, 1), filters=filters, kernel_size=kernel_size)
#convlay1 = Conv1D(64, 3, activation='relu', input_shape=(5, 1))
model.add(convlay1)
model.add(Activation('relu'))   # Function of ReLU activation is detector, [1], p. 71, fig. 5.9
poollay1 = MaxPooling1D(pool_size=2, strides=None, padding='valid')
model.add(poollay1)

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

              


model.fit(X_treino, np_utils.to_categorical(label_treina), batch_size=32, epochs=30, verbose=1)


fv = model.predict_classes(X_teste)
fv = fv.reshape((1,-1))[0]
classe = ['normal', 'falha']
funcoes.plot_confusion_matrix(confusion_matrix(label_tes,fv),classe)



#######################################################################


    
m3_w100 = funcoes.get_data('dados/m3_w100.mat','m3')
m3_w50 = funcoes.get_data('dados/m3_w50.mat','m3')
m3_w200 = funcoes.get_data('dados/m3_w200.mat','m3')
names_atributos = funcoes.get_data('dados/atributos_names.mat','class_name')
names_atributos = names_atributos[0]
teste = funcoes.get_data('200/teste_1.mat','m')


nomes_finais = []
for i in range(0,len(names_atributos)):
    
  names_atributos[i][0] = unicodedata.normalize('NFD', names_atributos[i][0]).encode('ascii', 'ignore')
  nomes_finais.append(names_atributos[i][0])
  

label_200 = m3_w200[:,-1]
m3_w200 = np.delete(m3_w200, -1, axis=1)

ada_selec = np.loadtxt('best_global_tudo25_ada.txt')
ada_selec = funcoes.excluir_coluna(ada_selec)



rf_rf_params_1 = np.loadtxt('best_global_tudo25_rf_params.txt')
rf_rf_params_1 = funcoes.excluir_coluna(rf_rf_params_1)

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

rf_params_2 = {
    
    'n_estimators': 1000,
    'bootstrap': True
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


names = ["RandomForestClassifier_1", "RandomForestClassifier_2","RandomForestClassifier_3",
         "AdaBoostClassifier", "keras"]

classifiers = [
    RandomForestClassifier(**rf_params_1),
    RandomForestClassifier(**rf_params),
    RandomForestClassifier(**rf_params_2),
    AdaBoostClassifier(),
    model                 
]
    

dici = {}
dici1 = {}
dici2 = {}
for name, clf in zip(names, classifiers):
        print 'Classificador: {}'.format(name)
        #clf.fit(m3_w100,label_100)
        #dici[name] = clf
        kfold = 15
        kf = StratifiedKFold(n_splits=kfold, random_state=42)
        #kf_10 = KFold(n_splits=kfold, random_state=42, shuffle=False)

        accuracy = []
        f1score = []
        if name != 'keras':
           fontP = FontProperties()
           fontP.set_size('small') 
           for i, (train_index, test_index) in enumerate(kf.split(m3_w200, label_200)):
           #for train_index, test_index in kf_10.split(m3_w200):    
              train_X, valid_X = m3_w200[train_index], m3_w200[test_index]
              train_y, valid_y = label_200[train_index], label_200[test_index]
              clf.fit(train_X,train_y)
              accuracy.append(accuracy_score(valid_y,clf.predict(valid_X)))
              f1score.append(f1_score(valid_y,clf.predict(valid_X),average='micro'))
              #print 'Confusion Matrix do classificador: {}'.format(name)
              #classe = ['normal', 'falha']
              #funcoes.plot_confusion_matrix(confusion_matrix(valid_y,clf.predict(valid_X)),classe)
              #accuracy.append(accuracy_score(valid_y,clf.predict(valid_X)))
              
              false_positive_rate, true_positive_rate, thresholds = roc_curve(valid_y,clf.predict(valid_X))
              roc_auc = auc(false_positive_rate, true_positive_rate)

              plt.title('Receiver Operating Characteristic')
              plt.plot(false_positive_rate, true_positive_rate, 'b',
              label='AUC = %0.2f'% roc_auc)
              plt.legend(loc='lower right',prop=fontP)
              plt.plot([0,1],[0,1],'r--')
              plt.xlim([-0.1,1.2])
              plt.ylim([-0.1,1.2])
              plt.ylabel('True Positive Rate')
              plt.xlabel('False Positive Rate')
              #plt.show()
        
           plt.show()
           clf.fit(m3_w200,label_200)
        else:
            fontP = FontProperties()
            fontP.set_size('small')
            X_treino = m3_w200.reshape(m3_w200.shape + (1,))
            for i, (train_index, test_index) in enumerate(kf.split(X_treino, label_200)):
            #for train_index, test_index in kf_10.split(m3_w200):    
                 train_X, valid_X = X_treino[train_index], X_treino[test_index]
                 train_y, valid_y = label_200[train_index], label_200[test_index]
                 clf.fit(train_X,train_y,batch_size=32, epochs=50)
                 accuracy.append(accuracy_score(valid_y,clf.predict_classes(valid_X)))
                 f1score.append(f1_score(valid_y,clf.predict_classes(valid_X),average='micro'))
                 #print 'Confusion Matrix do classificador: {}'.format(name)
                 #classe = ['normal', 'falha']
                 #funcoes.plot_confusion_matrix(confusion_matrix(valid_y,clf.predict(valid_X)),classe)
                 #accuracy.append(accuracy_score(valid_y,clf.predict(valid_X)))
                 false_positive_rate, true_positive_rate, thresholds = roc_curve(valid_y,clf.predict(valid_X))
                 roc_auc = auc(false_positive_rate, true_positive_rate)

                 plt.title('Receiver Operating Characteristic')
                 plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f'% roc_auc)
                 plt.legend(loc='lower right',prop=fontP)
                 plt.plot([0,1],[0,1],'r--')
                 plt.xlim([-0.1,1.2])
                 plt.ylim([-0.1,1.2])
                 plt.ylabel('True Positive Rate')
                 plt.xlabel('False Positive Rate')
                 #plt.show()
        
            plt.show()
            clf.fit(X_treino,label_200,batch_size=32, epochs=50)
        dici[name] = accuracy  
        dici2[name] = f1score    
        print 'Accuracia media do classificador: {}, kfold: {} - {}'.format(name,kfold,np.mean(accuracy))
        #clf.fit(m3_w200,label_200)
        dici1[name] = clf
         



'''avgDict = {}
for k,v in dici2.iteritems():
    # v is the list of grades for student k
    avgDict[k] = sum(v)/ float(len(v))
    
print 'Acuracia dos classificadores individuais: {}'.format(avgDict)
print 'Acuracia final: {}'.format(accuracy_score(label_tes, pred))

for k in dici1.keys():
   print 'Classificador: {}'.format(k) 
   clf = dici1[k]
   pred = clf.predict(teste)
   plt.plot(pred)
   plt.show()  '''
                        
x_train = np.array(m3_w200)
y_train = np.array(label_200)


#resultado = principal.iniciar(pd.DataFrame(m3_w200), label_200)

    
ntrain = x_train.shape[0]
SEED = 42
NFOLDS = 15 
#kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
kf = StratifiedKFold(n_splits=NFOLDS, random_state=SEED)
#kf = KFold(n_splits=kfold, random_state=42, shuffle=False)
    
rf_1 = Ensemble.SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params_1)
rf_2 = Ensemble.SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
rf_3 = Ensemble.SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params_2)
#et = Ensemble.SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params={})
ada = Ensemble.SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params={})
#gb = Ensemble.SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
rn = Ensemble.SklearnHelper(clf=model, seed=SEED,flag=5)

print('Iniciando o treinamento dos classificadores de primeiro nivel')
#print('Iniciando o ExtraTree')
#et_oof_train = funcoes.get_oof(et, x_train, y_train,ntrain, kf) # Extra Trees
   
print('Iniciando o RandomForest_1')
rf_oof_train= funcoes.get_oof(rf_1,x_train, y_train,ntrain, kf) # Random Forest
   
                        
print('Iniciando o RandomForest_2')
rf_oof_train_1= funcoes.get_oof(rf_2,x_train, y_train,ntrain, kf)


print('Iniciando o RandomForest_3')
rf_oof_train_3= funcoes.get_oof(rf_3,x_train, y_train,ntrain, kf)


print('Iniciando o Adaboost')
ada_oof_train = funcoes.get_oof(ada, x_train, y_train,ntrain, kf) # AdaBoost 
    
#X_train = preprocessing.scale(x_train)
X_train = x_train
X_train = X_train.reshape(X_train.shape + (1,))
print('Iniciando a Rede Convolucional')
rn_oof_train = funcoes.get_oof(rn,X_train, y_train,ntrain, kf)

#x_train_interm = np.concatenate((et_oof_train,rf_oof_train,rf_oof_train_1,ada_oof_train,gb_oof_train,rf_oof_train_3), axis=1)

x_train_interm = np.concatenate((rf_oof_train,rf_oof_train_1,ada_oof_train,rf_oof_train_3,rn_oof_train), axis=1)


X_treino, X_teste, label_treina, label_tes = train_test_split(x_train_interm,y_train,test_size=0.30, random_state=42)

'''gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
  }'''

print('Iniciando o treinamento do classificador de segundo nivel')

ada_final = AdaBoostClassifier()
ada_final.fit(X_treino,label_treina)
pred = ada_final.predict(X_teste)
classe = ['normal', 'falha']
funcoes.plot_confusion_matrix(confusion_matrix(label_tes,pred),classe)
           
           
################ todos os dados

print('Iniciando o RandomForest_1')
funcoes.get_oof_all_data(rf_1,x_train, y_train) # Random Forest
print('Iniciando o RandomForest_2')
funcoes.get_oof_all_data(rf_2,x_train, y_train)
print('Iniciando o RandomForest_3')
funcoes.get_oof_all_data(rf_3,x_train, y_train)
print('Iniciando o Adaboost')
funcoes.get_oof_all_data(ada, x_train, y_train) # AdaBoost 
#X_train = x_train.reshape(x_train.shape + (1,))
print('Iniciando a Rede Convolucional')
funcoes.get_oof_all_data(rn,X_train, y_train)

ada_final.fit(x_train_interm,y_train)



teste = funcoes.get_data('200/teste_1.mat','m')

dados_rf_1 = rf_1.predict_proba(teste) 
dados_rf_2 = rf_2.predict_proba(teste) 
dados_rf_3 = rf_3.predict_proba(teste) 
dados_ada =  ada.predict_proba(teste) 

X_testando = preprocessing.scale(teste)
       
dados_rn = rn.predict_proba(X_testando.reshape(X_testando.shape + (1,))) 

x_train_interm_tudo = np.concatenate((dados_rf_1,dados_rf_2,dados_rf_3,dados_ada,dados_rn), axis=1)
pred_final = ada_final.predict(x_train_interm_tudo)



for k in dici1.keys():
   print 'Classificador: {}'.format(k) 
   clf = dici1[k]
   pred = clf.predict(teste)
   plt.plot(pred)
   plt.savefig('dados/'+k+'.png')
   plt.show() 
   
   
plt.plot(pred_final)
plt.savefig('dados/ensemble.png')
plt.show()   



'''kf1 = StratifiedKFold(n_splits=NFOLDS, random_state=SEED)
for i, (train_index, test_index) in enumerate(kf1.split(x_train_interm, y_train)):
           x_tr = x_train_interm[train_index]
           y_tr = y_train[train_index]
           x_te = x_train_interm[test_index]
           y_te = y_train[test_index]  


           ada_final = AdaBoostClassifier()
           ada_final.fit(x_tr,y_tr)
           pred = ada_final.predict(x_te)

           classe = ['normal', 'falha']
           funcoes.plot_confusion_matrix(confusion_matrix(y_te,pred),classe)'''

avgDict = {}
for k,v in dici.iteritems():
    # v is the list of grades for student k
    avgDict[k] = sum(v)/ float(len(v))
    
print 'Acuracia dos classificadores individuais: {}'.format(avgDict)
print 'Acuracia final: {}'.format(accuracy_score(label_tes, pred))
 
classe = ['normal', 'falha']
funcoes.plot_confusion_matrix(confusion_matrix(label_tes,pred),classe)

'''pred_teste = ada_final.predict(teste)
plt.plot(pred)
plt.show()'''


'''for k in dici1.keys():
   print 'Classificador: {}'.format(k) 
   clf = dici1[k]
   pred = clf.predict(teste)
   plt.plot(pred)
   plt.show()  ''' 

  
''' collections import OrderedDict
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt"
                               )),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 15
max_estimators = 1000

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(m3_w100, label_100)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()'''




    
    
    
import xgboost as xgb   

xgb_params_1 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'silent': True,
    'num_class': 2 
     
                 }

xgb_params_2 = {
    'eta': 0.02,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'silent': True 
                 }


X_treino, X_teste, label_treina, label_tes = train_test_split(m3_w100,label_100,test_size=0.20, random_state=42)

d_train = xgb.DMatrix(X_treino, label_treina)
d_valid = xgb.DMatrix(X_teste, label_tes)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
model = xgb.train(xgb_params_2, d_train, 5000,  watchlist, maximize=True, verbose_eval=50, early_stopping_rounds=100)

d_teste = xgb.DMatrix(X_teste)

pred = model.predict(d_teste)
#classe = ['normal', 'falha']
#funcoes.plot_confusion_matrix(confusion_matrix(label_tes,pred),classe)

  


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
#from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical 
from keras.layers import Embedding



####################################


from sklearn import preprocessing
             
X_treino, X_teste, label_treina, label_tes = train_test_split(m3_w200,label_200,test_size=0.20, random_state=42)

X_treino = preprocessing.scale(X_treino)
X_teste = preprocessing.scale(X_teste)

X_treino = X_treino.reshape(X_treino.shape + (1,))
X_teste = X_teste.reshape(X_teste.shape + (1,))



             
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(55, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_treino, np_utils.to_categorical(label_treina)
, batch_size=32, epochs=50, verbose=0)
#score = model.evaluate(X_teste, label_tes, batch_size=32)


             
fv = model.predict_classes(X_teste)
fv = fv.reshape((1,-1))[0]
classe = ['normal', 'falha']
funcoes.plot_confusion_matrix(confusion_matrix(label_tes,fv),classe)



testando = preprocessing.scale(teste)
testando = testando.reshape(testando.shape + (1,))

    

'''kf_10 = KFold(n_splits=10, random_state=None, shuffle=False)

for train_index, test_index in kf_10.split(m3_w200):
    
    X_train, X_test = x_train[train_index], x_train[test_index]
    y_train_10, y_test_10 = y_train[train_index], y_train[test_index]
    
    
    print 'distribuicao no treinamento: \n{}'.format(pd.value_counts(y_train_10))
    print 'distribuicao no teste: \n{} \n'.format(pd.value_counts(y_test_10))'''
    
    
    