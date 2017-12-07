# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 11:52:07 2017

@author: rebli
"""
'''import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev0\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']'''

import numpy as np
import pandas as pd





import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
import xgboost as xgb
import Principal as principal
reload(principal)
from sklearn import model_selection, preprocessing
import funcoes as funcoes
reload(funcoes)
from itertools import combinations

def excluir_coluna(s):
        lista = []
           
        for i in range(0,len(s)-1):
               
               if s[i] > 0.6:
                  lista.append(i)
                  
        return lista


          
# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
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



train = pd.read_csv("train/train.csv", na_values=-1)
teste = pd.read_csv("test/test.csv", na_values=-1)
sub = pd.read_csv("sample_submission/sample_submission.csv",na_values=-1)


#ind, = np.where(target_train == 0)
#ind1, = np.where(target_train == 1)
#ulimit = np.percentile(train.ps_car_13.values, 99.7)
#llimit = np.percentile(train.ps_car_13.values, 1)
#superior = train[train.ps_car_13 > ulimit].index
#inferior = train[train.ps_car_13 < llimit].index
#train = train.drop(superior, axis=0)
#train = train.drop(inferior, axis=0)  

target_train = train['target'].values
id_test = teste['id'].values               
train=train.drop(['target','id'],axis=1)
teste=teste.drop(['id'], axis = 1)

selec = np.loadtxt('best_global_tudo3.txt')
selec = excluir_coluna(selec)

train = train.iloc[:,selec]
teste = teste.iloc[:,selec]




for c, dtype in zip(train.columns, train.dtypes):
    if dtype == np.float64:
        train[c] = train[c].astype(np.float32)
        teste[c] = teste[c].astype(np.float32)
    elif  dtype == np.int64:
        train[c] = train[c].astype(np.int32)
        teste[c] = teste[c].astype(np.int32)




'''traintest = pd.concat([train, teste], axis = 0)
feats_counts = train.nunique(dropna = False)

feats_counts.sort_values()[:10]
   
  
dup_cols = {}

for i, c1 in enumerate(train.columns):
    for c2 in train.columns[i + 1:]:
        if c2 not in dup_cols and np.all(train[c1] == train[c2]):
            dup_cols[c2] = c1   
   
def autolabel(arrayA):
    ''' label each colored square with the corresponding data value. 
    If value > 20, the text is in black, else in white.
    '''
    arrayA = np.array(arrayA)
    for i in range(arrayA.shape[0]):
        for j in range(arrayA.shape[1]):
                plt.text(j,i, "%.2f"%arrayA[i,j], ha='center', va='bottom',color='w')
                
                
def gt_matrix(feats,sz=16):
    a = []
    for i,c1 in enumerate(feats):
        b = [] 
        for j,c2 in enumerate(feats):
            mask = (~feats[c1].isnull()) & (~feats[c2].isnull())
            if i>=j:
                b.append((feats.loc[mask,c1].values>=feats.loc[mask,c2].values).mean())
            else:
                b.append((feats.loc[mask,c1].values>feats.loc[mask,c2].values).mean())

        a.append(b)

    plt.figure(figsize = (sz,sz))
    plt.imshow(a, interpolation = 'None')
    _ = plt.xticks(range(len(feats)),feats,rotation = 90)
    _ = plt.yticks(range(len(feats)),feats,rotation = 0)
    autolabel(a)     



feats = train.iloc[:,:20]
# build 'mean(feat1 > feat2)' plot
gt_matrix(feats,16)  '''





    
##################################
xgb_params_2 = {
    'eta': 0.02,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'silent': True
     
                 }

'''train['target'] = target_train
dicionario_res = {}
import itertools   
for i in range(1,7):
    
   gb = pd.value_counts(train.iloc[:,0])
   lista = list(gb.keys())
   gh = list(itertools.combinations(lista,i))
   for j in gh:
       
    sel = list(j)
    ind = train[train.iloc[:,0].isin(sel)].index
                
    treino_1 = train.loc[ind]
    y_trein_1 = treino_1.target
    treino_1 = treino_1.drop('target', axis=1)
    
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(treino_1, y_trein_1, test_size=0.2, random_state=42)

    d_train_1 = xgb.DMatrix(X_train_1, y_train_1)
    d_valid_1 = xgb.DMatrix(X_test_1, y_test_1)
    watchlist_1 = [(d_train_1, 'train'), (d_valid_1, 'valid')]'''
    
    '''treino_2 = train.loc[list(set(train.index) - set(ind))]
    y_trein_2 = treino_2.target
    treino_2 = treino_2.drop('target', axis=1)
    
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(treino_2, y_trein_2, test_size=0.2, random_state=42)

    d_train_2 = xgb.DMatrix(X_train_2, y_train_2)
    d_valid_2 = xgb.DMatrix(X_test_2, y_test_2)
    watchlist_2 = [(d_train_2, 'train'), (d_valid_2, 'valid')]'''
    
    
    
    #model_train_1 = xgb.train(xgb_params_2, d_train_1, 5000,  watchlist_1, maximize=True, verbose_eval=False,feval=gini_xgb, early_stopping_rounds=100)
    #model_train_2 = xgb.train(xgb_params_2, d_train_2, 5000,  watchlist_2, maximize=True, verbose_eval=False,feval=gini_xgb, early_stopping_rounds=100)
    #print 'Model_train_1_gini: {}, Valores: {}'.format(model_train_1.best_score,sel)
    #print 'Model_train_2_gini: {}'.format(model_train_2.best_score)
    #print '#############################################################'
    #num_boost_rounds = model1.best_iteration
    #d_train = xgb.DMatrix(train, target_train)

    #clf = xgb.train(xgb_params_2, d_train, num_boost_round=num_boost_rounds,feval=gini_xgb, verbose_eval=10) 
        
        
        
        
        
    
    

'''d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
dcol = [c for c in train.columns if c not in ['id','target']]
train['ps_car_13_x_ps_reg_03'] = train['ps_car_13'] * train['ps_reg_03']
train['negative_one_vals'] = np.sum((train[dcol]==-1).values, axis=1)
for c in dcol:
  if '_bin' not in c:
      train[c+str('_median_range')] = (train[c].values > d_median[c]).astype(np.int)
      train[c+str('_mean_range')] = (train[c].values > d_mean[c]).astype(np.int)

    
dcol = [c for c in teste.columns if c not in ['id','target']]
teste['ps_car_13_x_ps_reg_03'] = teste['ps_car_13'] * teste['ps_reg_03']
teste['negative_one_vals'] = np.sum((teste[dcol]==-1).values, axis=1)
for c in dcol:
  if '_bin' not in c:
      teste[c+str('_median_range')] = (teste[c].values > d_median[c]).astype(np.int)
      teste[c+str('_mean_range')] = (teste[c].values > d_mean[c]).astype(np.int)'''


'''combine= pd.concat([train,teste],axis=0)

# Performing one hot encoding
cat_features = [a for a in combine.columns if a.endswith('cat')]
for column in cat_features:
	temp=pd.get_dummies(pd.Series(combine[column]))
	combine=pd.concat([combine,temp],axis=1)
	combine=combine.drop([column],axis=1)

train=combine[:train.shape[0]]
teste=combine[train.shape[0]:]'''

train = np.array(train)
teste = np.array(teste)
#print ("The train shape is:",train.shape)
#print ('The test shape is:',teste.shape)

xgb_preds = []

'''xgb_params_1 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'silent': True
     
                 }'''

#X_train, X_test, y_train, y_test = train_test_split(train, target_train, test_size=0.2, random_state=42)
#d_train = xgb.DMatrix(X_train, y_train)
#d_valid = xgb.DMatrix(X_test, y_test)
'''watchlist = [(d_train, 'train'), (d_valid, 'valid')]
model = xgb.train(xgb_params_1, d_train, 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=100)
num_boost_rounds = model.best_iteration
d_train = xgb.DMatrix(train, target_train)

clf = xgb.train(xgb_params_1, d_train, num_boost_round=num_boost_rounds,feval=gini_xgb, verbose_eval=10)    '''

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
    'silent': True
     
                 }



'''for i in train.columns:
    name_1 = 'var_left_' + str(i) 
    for j in train.columns:
      interm = 'var_right_' + str(j)
      name_2 = name_1 +  interm
      train[name_2] = train[i] * train[j]'''
        

'''nome = 'ps_ind_01'
for i in train.columns:
    
    name_2 = nome + "_" +  str(i)
    train[name_2] = train[nome] * train[i]'''
    
    
'''nome = 'ps_ind_01'
indices = np.arange(train.shape[0])
train1 = train.copy()
for i in train.columns:
    
    name_2 = nome + "_" +  str(i)
    print('nome:', name_2)
    train[name_2] = train[nome] * train[i]
    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(train, target_train, indices, test_size=0.2, random_state=42)
    
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_test, y_test)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    model1 = xgb.train(xgb_params_2, d_train, 5000,  watchlist, maximize=True, verbose_eval=50,feval=gini_xgb, early_stopping_rounds=100)
    train = train1.copy()'''
#target_train = train.target
#train = train.drop('target', axis=1)



#indices = np.arange(train.shape[0])
#X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(train, target_train, indices, test_size=0.2, random_state=42)


##############################################################3



lista = ['ps_ind_01_ps_ind_07_bin','ps_ind_01_ps_ind_08_bin']

for i in lista:    
   lbl = preprocessing.LabelEncoder()
   lbl.fit(list(train[i].values)) 
   train[i] = lbl.transform(list(train[i].values))
   teste[i] = lbl.transform(list(teste[i].values))
   
   
nome = 'ps_ind_03'
for i in train.columns[0:24]:
    name_2 = nome + '_' + str(i)
    train[name_2] = train[nome] * train[i]
    
nome = 'ps_ind_03'
for i in teste.columns[0:24]:
    name_2 = nome + '_' + str(i)
    teste[name_2] = teste[nome] * teste[i]    
    
selec = np.loadtxt('best_global_tudo15.txt')
selec = excluir_coluna(selec)

train = train[selec]
teste = teste[selec]    

all_objects = neigh.kneighbors(train.fillna(-999), return_distance=False)

'''train['target'] = target_train
teste['id'] = id_test     
train_1 = train[train.ps_car_13 > 1.7]
teste_1 = teste[teste.ps_car_13 > 1.7]
train_2 = train[train.ps_car_13 <= 1.7]
teste_2 = teste[teste.ps_car_13 <= 1.7]'''


##########################################
#train['perce_0'] = perce_0
#train['perce_1'] = perce_1

#teste['perce_0'] = perce_0_teste
#teste['perce_1'] = perce_1_teste

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

     

traintest = pd.concat([train, teste], axis = 0)
fodase = pd.value_counts(traintest.ps_car_13)
testando = lambda x: fodase[x]
traintest['first'] = traintest.ps_car_13.apply(testando)

encoding = traintest.groupby('ps_ind_03').size()
encoding = encoding / len(traintest)
traintest['ps_ind_03_freq'] = traintest.ps_ind_03.map(encoding)


'''def funcao(x):
    x1 = x['ps_car_11'].fillna(0.0)
    x2 = x['ps_reg_01'].fillna(0.0)
    gb.loc[x1, x2] 
    return x1'''
    
gb = traintest.groupby('ps_car_11')['ps_reg_01'].value_counts().unstack().fillna(0)
funcao = lambda x: gb.loc[x.ps_car_11, x.ps_reg_01]    
traintest['freq_01'] = traintest.fillna(0).apply(funcao, axis=1) 

'''gb = traintest.groupby('ps_car_11')['ps_reg_02'].value_counts().unstack().fillna(0)
funcao = lambda x: gb.loc[x.ps_car_11, x.ps_reg_02]    
traintest['freq_02'] = traintest.fillna(0).apply(funcao, axis=1)'''

'''gb = traintest.groupby('ps_car_11')['ps_ind_01'].value_counts().unstack().fillna(0)
funcao = lambda x: gb.loc[x.ps_car_11, x.ps_ind_01]    
traintest['freq_03'] = traintest.fillna(0).apply(funcao, axis=1)'''

'''gb = traintest.groupby('ps_car_11')['ps_ind_05_cat'].value_counts().unstack().fillna(0)
funcao = lambda x: gb.loc[x.ps_car_11, x.ps_ind_05_cat]    
traintest['freq_04'] = traintest.fillna(0).apply(funcao, axis=1) '''

gb = traintest.groupby('ps_car_11')['ps_car_11_cat'].value_counts().unstack().fillna(0)
funcao = lambda x: gb.loc[x.ps_car_11, x.ps_car_11_cat]    
traintest['freq_05'] = traintest.fillna(0).apply(funcao, axis=1) 

gb = traintest.groupby('ps_car_11')['ps_ind_05_cat'].value_counts().unstack().fillna(0)
funcao = lambda x: gb.loc[x.ps_car_11, x.ps_ind_05_cat]    
traintest['freq_06'] = traintest.fillna(0).apply(funcao, axis=1) 

'''gb = traintest.groupby('ps_car_11')['ps_ind_15'].value_counts().unstack().fillna(0)
funcao = lambda x: gb.loc[x.ps_car_11, x.ps_ind_15]    
traintest['freq_07'] = traintest.fillna(0).apply(funcao, axis=1) '''


'''gb = traintest.groupby('ps_car_11')['ps_car_06_cat'].value_counts().unstack().fillna(0)
funcao = lambda x: gb.loc[x.ps_car_11, x.ps_car_06_cat]    
traintest['freq_08'] = traintest.fillna(0).apply(funcao, axis=1) '''

traintest['ps_ind_01_ps_ind_07_bin'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_07_bin'].map(str)).astype(int)
#traintest['ps_car_13_ps_ind_07_bin'] = traintest['ps_ind_01'].map(str)+traintest['ps_ind_07_bin'].map(str)

traintest['ps_ind_01_ps_ind_03'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_03'].map(str)).astype(int)

#traintest['ps_ind_01_ps_ind_14'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_14'].map(str)).astype(int)

traintest['ps_ind_01_ps_ind_15'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_15'].map(str)).astype(int)


#traintest['ps_ind_01_ps_ind_06_bin'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_06_bin'].map(str)).astype(int)

traintest['ps_ind_01_ps_ind_08_bin'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_08_bin'].map(str)).astype(int)

#traintest['ps_ind_01_ps_ind_09_bin'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_09_bin'].map(str)).astype(int)

traintest['ps_ind_01_ps_ind_10_bin'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_10_bin'].map(str)).astype(int)

#traintest['ps_ind_01_ps_ind_11_bin'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_11_bin'].map(str)).astype(int)

#traintest['ps_ind_01_ps_ind_12_bin'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_12_bin'].map(str)).astype(int)

#traintest['ps_ind_01_ps_ind_13_bin'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_13_bin'].map(str)).astype(int)

#traintest['ps_ind_01_ps_ind_16_bin'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_16_bin'].map(str)).astype(int)

#traintest['ps_ind_01_ps_ind_16_bin'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_16_bin'].map(str)).astype(int)

#traintest['ps_ind_01_ps_ind_17_bin'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_17_bin'].map(str)).astype(int)
#traintest['ps_ind_01_ps_ind_18_bin'] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_18_bin'].map(str)).astype(int)

#traintest['ps_ind_01_ps_car_01_cat'] = (traintest['ps_ind_01'].map(str)+traintest['ps_car_01_cat'].map(str))

#traintest['ps_ind_01_ps_car_03_cat'] = (traintest['ps_ind_01'].map(str)+traintest['ps_car_03_cat'].map(str))

#traintest['ps_ind_03_ps_ind_15'] = (traintest['ps_ind_03'].map(str)+traintest['ps_ind_15'].map(str))

#traintest['ps_ind_03_ps_ind_14'] = (traintest['ps_ind_03'].map(str)+traintest['ps_ind_14'].map(str))

#traintest['ps_ind_03_ps_ind_11_bin'] = (traintest['ps_ind_03'].map(str)+traintest['ps_ind_11_bin'].map(str))

#traintest['ps_ind_03_ps_ind_12_bin'] = (traintest['ps_ind_03'].map(str)+traintest['ps_ind_12_bin'].map(str))

#traintest['ps_ind_03_ps_ind_13_bin'] = (traintest['ps_ind_03'].map(str)+traintest['ps_ind_13_bin'].map(str))

#traintest['ps_ind_03_ps_car_01_cat'] = (traintest['ps_ind_03'].map(str)+traintest['ps_car_01_cat'].map(str))

#traintest['ps_ind_06_binxps_ind_07_binzps_ind_08_bin'] = (traintest['ps_ind_06_bin'].map(str)+traintest['ps_ind_07_bin'].map(str)+traintest['ps_ind_08_bin'].map(str))
nome10 = 'ps_ind_02_cat' + '_' + 'ps_ind_03' + '_' + 'ps_ind_06_bin' + '_' + 'ps_ind_08_bin'
traintest[nome10] = (traintest['ps_ind_02_cat'].map(str)+traintest['ps_ind_03'].map(str)+traintest['ps_ind_06_bin'].map(str)+traintest['ps_ind_08_bin'].map(str))

nome11 = 'ps_ind_01' + '_' + 'ps_ind_07_bin' + '_' + 'ps_ind_09_bin' + '_' + 'ps_ind_12_bin'
traintest[nome11] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_07_bin'].map(str)+traintest['ps_ind_09_bin'].map(str)+traintest['ps_ind_12_bin'].map(str))

'ps_ind_06_bin_ps_ind_09_bin_ps_ind_12_bin_ps_ind_13_bin'

nome12 = 'ps_ind_06_bin' + '_' + 'ps_ind_09_bin' + '_' + 'ps_ind_12_bin' + '_' + 'ps_ind_13_bin'
traintest[nome12] = (traintest['ps_ind_06_bin'].map(str)+traintest['ps_ind_09_bin'].map(str)+traintest['ps_ind_12_bin'].map(str)+traintest['ps_ind_13_bin'].map(str))

nome13 = 'ps_ind_03' + '_' + 'ps_ind_05_cat' + '_' + 'ps_ind_09_bin' + '_' + 'ps_ind_10_bin'
traintest[nome13] = (traintest['ps_ind_03'].map(str)+traintest['ps_ind_05_cat'].map(str)+traintest['ps_ind_09_bin'].map(str)+traintest['ps_ind_10_bin'].map(str))


nome14 = 'ps_ind_03' + '_' + 'ps_ind_05_cat' + '_' + 'ps_ind_07_bin' + '_' + 'ps_ind_08_bin'
traintest[nome14] = (traintest['ps_ind_03'].map(str)+traintest['ps_ind_05_cat'].map(str)+traintest['ps_ind_07_bin'].map(str)+traintest['ps_ind_08_bin'].map(str))
#nome12 = 'ps_ind_01' + '_' + 'ps_ind_09_bin' + '_' + 'ps_ind_11_bin' + '_' + 'ps_ind_13_bin'
#traintest[nome12] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_09_bin'].map(str)+traintest['ps_ind_11_bin'].map(str)+traintest['ps_ind_13_bin'].map(str))
#ps_ind_01_ps_ind_04_cat_ps_ind_09_bin_ps_ind_13_bin


nome15 = 'ps_ind_04_cat' + '_' + 'ps_ind_07_bin' + '_' + 'ps_ind_08_bin' + '_' + 'ps_ind_10_bin' + 'ps_ind_11_bin'
traintest[nome15] = (traintest['ps_ind_04_cat'].map(str)+traintest['ps_ind_07_bin'].map(str)+traintest['ps_ind_08_bin'].map(str)+traintest['ps_ind_10_bin'].map(str)+traintest['ps_ind_11_bin'].map(str))
#nome15 = 'ps_ind_03' + '_' + 'ps_ind_04_cat' + '_' + 'ps_ind_09_bin' + '_' + 'ps_ind_13_bin'
#traintest[nome15] = (traintest['ps_ind_03'].map(str)+traintest['ps_ind_04_cat'].map(str)+traintest['ps_ind_09_bin'].map(str)+traintest['ps_ind_13_bin'].map(str))


nome16 = 'ps_reg_02' + '_' + 'ps_car_06_cat' + '_' + 'ps_ind_01_ps_ind_15'
traintest[nome16] = (traintest['ps_reg_02'].map(str)+traintest['ps_car_06_cat'].map(str)+traintest['ps_ind_01_ps_ind_15'].map(str))


nome17 = 'ps_ind_15' + '_' + 'ps_car_06_cat' + '_' + 'ps_ind_03_ps_ind_05_cat_ps_ind_07_bin_ps_ind_08_bin'
traintest[nome17] = (traintest['ps_ind_15'].map(str)+traintest['ps_car_06_cat'].map(str)+traintest['ps_ind_03_ps_ind_05_cat_ps_ind_07_bin_ps_ind_08_bin'].map(str))


####



nome18 = 'ps_car_11_cat' + '_' + 'ps_ind_01_ps_ind_15' + '_' + 'ps_ind_01_ps_ind_07_bin_ps_ind_09_bin_ps_ind_12_bin'
traintest[nome18] = (traintest['ps_car_11_cat'].map(str)+traintest['ps_ind_01_ps_ind_15'].map(str)+traintest['ps_ind_01_ps_ind_07_bin_ps_ind_09_bin_ps_ind_12_bin'].map(str))

nome19 = 'ps_car_05_cat' + '_' + 'first' + '_' + 'ps_ind_03_ps_ind_05_cat_ps_ind_09_bin_ps_ind_10_bin'
traintest[nome19] = (traintest['ps_car_05_cat'].map(str)+traintest['first'].map(str)+traintest['ps_ind_03_ps_ind_05_cat_ps_ind_09_bin_ps_ind_10_bin'].map(str))

nome20 = 'ps_ind_02_cat' + '_' + 'ps_ind_08_bin' + '_' + 'ps_ind_18_bin' + '_' + 'ps_ind_15_ps_car_06_cat_ps_ind_03_ps_ind_05_cat_ps_ind_07_bin_ps_ind_08_bin'
traintest[nome20] = (traintest['ps_ind_02_cat'].map(str)+traintest['ps_ind_08_bin'].map(str)+traintest['ps_ind_18_bin'].map(str)+traintest['ps_ind_15_ps_car_06_cat_ps_ind_03_ps_ind_05_cat_ps_ind_07_bin_ps_ind_08_bin'].map(str))

  

nome21 = 'ps_ind_05_cat' + '_' + 'ps_car_07_cat' + '_' + 'ps_ind_01_ps_ind_15' + '_' + 'ps_reg_02_ps_car_06_cat_ps_ind_01_ps_ind_15'
traintest[nome21] = (traintest['ps_ind_05_cat'].map(str)+traintest['ps_car_07_cat'].map(str)+traintest['ps_ind_01_ps_ind_15'].map(str)+traintest['ps_reg_02_ps_car_06_cat_ps_ind_01_ps_ind_15'].map(str))

 
nome22 = 'ps_ind_05_cat' + '_' + 'ps_ind_10_bin' + '_' + 'ps_reg_02' + '_' + 'ps_reg_02_ps_car_06_cat_ps_ind_01_ps_ind_15'
traintest[nome22] = (traintest['ps_ind_05_cat'].map(str)+traintest['ps_ind_10_bin'].map(str)+traintest['ps_reg_02'].map(str)+traintest['ps_reg_02_ps_car_06_cat_ps_ind_01_ps_ind_15'].map(str))


 

'''nome23 = 'ps_car_05_cat' + '_' + 'freq_05' + '_' + 'ps_ind_01_ps_ind_07_bin_ps_ind_09_bin_ps_ind_12_bin' + '_' + 'ps_ind_06_bin_ps_ind_09_bin_ps_ind_12_bin_ps_ind_13_bin'
traintest[nome23] = (traintest['ps_car_05_cat'].map(str)+traintest['freq_05'].map(str)+traintest['ps_ind_01_ps_ind_07_bin_ps_ind_09_bin_ps_ind_12_bin'].map(str)+traintest['ps_ind_06_bin_ps_ind_09_bin_ps_ind_12_bin_ps_ind_13_bin'].map(str))  '''       
         

'''nome23 = 'ps_ind_01' + '_' + 'ps_car_06_cat' + '_' + 'ps_ind_01_ps_ind_07_bin' + '_' + 'ps_ind_15_ps_car_06_cat_ps_ind_03_ps_ind_05_cat_ps_ind_07_bin_ps_ind_08_bin'
traintest[nome23] = (traintest['ps_ind_01'].map(str)+traintest['ps_car_06_cat'].map(str)+traintest['ps_ind_01_ps_ind_07_bin'].map(str)+traintest['ps_ind_15_ps_car_06_cat_ps_ind_03_ps_ind_05_cat_ps_ind_07_bin_ps_ind_08_bin'].map(str))'''        

         
'''nome23 = 'ps_ind_01' + '_' + 'ps_ind_02_cat' + '_' + 'freq_05' + '_' + 'ps_ind_01_ps_ind_10_bin'
traintest[nome23] = (traintest['ps_ind_01'].map(str)+traintest['ps_ind_02_cat'].map(str)+traintest['freq_05'].map(str)+traintest['ps_ind_01_ps_ind_10_bin'].map(str))'''  

 
'''nome23 = 'ps_ind_03' + '_' + 'ps_car_04_cat' + '_' + 'ps_ind_01_ps_ind_10_bin' + '_' + 'ps_ind_15_ps_car_06_cat_ps_ind_03_ps_ind_05_cat_ps_ind_07_bin_ps_ind_08_bin'
traintest[nome23] = (traintest['ps_ind_03'].map(str)+traintest['ps_car_04_cat'].map(str)+traintest['ps_ind_01_ps_ind_10_bin'].map(str)+traintest['ps_ind_15_ps_car_06_cat_ps_ind_03_ps_ind_05_cat_ps_ind_07_bin_ps_ind_08_bin'].map(str))'''

         
         
drop = ['ps_calc_01','ps_calc_02','ps_calc_06','ps_calc_08','ps_calc_09','ps_calc_10',
        'ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin',
        'ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']


traintest.drop(drop, axis=1, inplace=True)
#nome14 = 'ps_ind_03' + '_' + 'ps_ind_05_cat' + '_' + 'ps_ind_07_bin' + '_' + 'ps_ind_08_bin'
#traintest[nome14] = (traintest['ps_ind_03'].map(str)+traintest['ps_ind_05_cat'].map(str)+traintest['ps_ind_07_bin'].map(str)+traintest['ps_ind_08_bin'].map(str))
    

#nome15 = 'ps_ind_03' + '_' + 'ps_ind_05_cat' + '_' + 'ps_ind_07_bin' + '_' + 'ps_ind_09_bin'
#traintest[nome15] = (traintest['ps_ind_03'].map(str)+traintest['ps_ind_05_cat'].map(str)+traintest['ps_ind_07_bin'].map(str)+traintest['ps_ind_09_bin'].map(str))

#nome1 = 'ps_ind_03' + '_' + 'ps_ind_08_bin' + '_' + 'ps_ind_09_bin' + '_' + 'ps_ind_12_bin'
#traintest[nome1] = (traintest['ps_ind_03'].map(str)+traintest['ps_ind_08_bin'].map(str)+traintest['ps_ind_09_bin'].map(str)+traintest['ps_ind_12_bin'].map(str))

#nome1 = 'ps_ind_03' + '_' + 'ps_ind_04_cat' + '_' + 'ps_ind_05_cat' + '_' + 'ps_ind_09_bin'
#traintest[nome1] = (traintest['ps_ind_03'].map(str)+traintest['ps_ind_04_cat'].map(str)+traintest['ps_ind_05_cat'].map(str)+traintest['ps_ind_09_bin'].map(str))

columnsToEncode = ['ps_ind_01_ps_ind_07_bin', 'ps_ind_01_ps_ind_03','ps_ind_01_ps_ind_15','ps_ind_01_ps_ind_08_bin','ps_ind_01_ps_ind_10_bin',
                   nome15]
                      


columnsToEncode1 = [nome10,nome11,nome12,nome13,nome14,nome16,nome17,nome18,nome19,nome20,
                    nome21,nome22] 

                    

#columnsToEncode1 = [nome10,nome11,nome12,nome13,nome14,nome16]                        

for i in columnsToEncode:
   qtd = pd.value_counts(traintest[i])
   testando = lambda x: qtd[x]
   traintest[i] = traintest[i].apply(testando)

 

for c in columnsToEncode1:
    
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(traintest[c].values)) 
        traintest[c] = lbl.transform(list(traintest[c].values))
        
                 
                 
                 






train = traintest.iloc[:train.shape[0],:]
teste = traintest.iloc[train.shape[0]:,:]

ntrain = train.shape[0]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
feature = funcoes.get_oof_encoding(train, target_train,ntrain, kf, nome16)
train[nome16] = feature

feature = funcoes.get_oof_encoding(train, target_train,ntrain, kf, nome17)
train[nome17] = feature       
        
feature = funcoes.get_oof_encoding(train, target_train,ntrain, kf, nome18)
train[nome18] = feature

feature = funcoes.get_oof_encoding(train, target_train,ntrain, kf, nome19)
train[nome19] = feature 


feature = funcoes.get_oof_encoding(train, target_train,ntrain, kf, nome20)
train[nome20] = feature 

feature = funcoes.get_oof_encoding(train, target_train,ntrain, kf, nome21)
train[nome21] = feature 

feature = funcoes.get_oof_encoding(train, target_train,ntrain, kf, nome22)
train[nome22] = feature


     

'''feature = funcoes.get_oof_encoding(train, target_train,ntrain, kf, nome23)
train[nome23] = feature  '''  
      
train['target'] = target_train
     
null, teste[nome16] = funcoes.target_encode(train[nome16], 
                         teste[nome16], 
                         target=train.target, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)

null, teste[nome17] = funcoes.target_encode(train[nome17], 
                         teste[nome17], 
                         target=train.target, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)


null, teste[nome18] = funcoes.target_encode(train[nome18], 
                         teste[nome18], 
                         target=train.target, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)

null, teste[nome19] = funcoes.target_encode(train[nome19], 
                         teste[nome19], 
                         target=train.target, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)


null, teste[nome20] = funcoes.target_encode(train[nome20], 
                         teste[nome20], 
                         target=train.target, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)


null, teste[nome21] = funcoes.target_encode(train[nome21], 
                         teste[nome21], 
                         target=train.target, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)


null, teste[nome22] = funcoes.target_encode(train[nome22], 
                         teste[nome22], 
                         target=train.target, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)


train.drop('target', axis=1, inplace=True)
  

#######################





#trn_df - treino
#sub_df - teste
# Target encode ps_car_11_cat
'''train['target_train'] = target_train
train_col_1, teste_col_1 = funcoes.target_encode(train["ps_car_11_cat"], 
                         teste["ps_car_11_cat"], 
                         target=train.target_train, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)'''



'''train_col_2, teste_col_2 = funcoes.target_encode(train["ps_ind_02_cat_ps_ind_03_ps_ind_06_bin_ps_ind_08_bin"], 
                         teste["ps_ind_02_cat_ps_ind_03_ps_ind_06_bin_ps_ind_08_bin"], 
                         target=train.target_train, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)'''

#feature = funcoes.get_oof_encoding(train, target_train,ntrain, kf, 'ps_car_13_ps_ind_02_cat_ps_ind_03_ps_ind_06_bin_ps_ind_08_bin')
#train['ps_car_13_ps_ind_02_cat_ps_ind_03_ps_ind_06_bin_ps_ind_08_bin'] = feature                                             
                                                    
#train.drop('target_train', axis=1, inplace=True)

#train['testando_1'] = train_col_1
#teste['testando_1'] = teste_col_1
     
#train['testando_2'] = train_col_2
#teste['testando_2'] = teste_col_2

#train.drop(['ps_car_11_cat'], axis=1, inplace=True)     
#teste.drop(['ps_car_11_cat'], axis=1, inplace=True)     

#x_train = np.array(train)
x_train = train.copy()
kfold = 3
#kf = StratifiedKFold(n_splits=kfold, random_state=42)
random_state = [42,43,44,45,46]
xgb_bold_final2 = {}

for k in random_state:
  print 'Random state: {}'.format(k)  
  xgb_bold_varios52 = []
  
  
  kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=k)   
  for p, (train_index, test_index) in enumerate(kf.split(x_train, target_train)):
      train_X, valid_X = x_train.iloc[train_index], x_train.iloc[test_index]
      train_y, valid_y = target_train[train_index], target_train[test_index]
      
      


      d_train = xgb.DMatrix(train_X, train_y)
      d_valid = xgb.DMatrix(valid_X, valid_y)
    
    
      watchlist = [(d_train, 'train'), (d_valid, 'valid')]

      model = xgb.train(xgb_params_2, d_train, 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=100)
      xgb_bold_varios52.append(model.best_score)  
     
  xgb_bold_final2[k] = xgb_bold_varios52

############################################################################3


kfold = 5
xgb_bold_final1 = {}

colunas = list(train.columns)

shuffle(colunas)

for i in colunas[20:]:
    print(i)
    #train1 = train.copy()
    #teste1 = teste.copy()
    
        
    #teste.drop('ps_car_11_cat', axis=1, inplace=True)  
    x_train = train.copy()
    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    xgb_bold_varios45 = []
    for p, (train_index, test_index) in enumerate(kf.split(x_train, target_train)):
      train_X, valid_X = x_train.iloc[train_index], x_train.iloc[test_index]
      train_y, valid_y = target_train[train_index], target_train[test_index]

    
      train_X['target_train'] = train_y
      train_col_1, teste_col_1 = funcoes.target_encode(train_X[i], 
                         valid_X[i], 
                         target=train_X.target_train, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)
      
      train_X.drop('target_train', axis=1, inplace=True)
      
      train_X['testando_1'] = train_col_1
      valid_X['testando_1'] = teste_col_1
     

      train_X.drop([i], axis=1, inplace=True)     
      valid_X.drop([i], axis=1, inplace=True) 
    
    
    
      d_train = xgb.DMatrix(train_X, train_y)
      d_valid = xgb.DMatrix(valid_X, valid_y)
    
      #d_test = xgb.DMatrix(teste)
    
      watchlist = [(d_train, 'train'), (d_valid, 'valid')]
      #clf = xgb.train(xgb_params_2, d_train, num_boost_round=5000,feval=gini_xgb, verbose_eval=50)    

      model = xgb.train(xgb_params_2, d_train, 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=False, early_stopping_rounds=100)
      xgb_bold_varios45.append(model.best_score)  
      
    xgb_bold_final[i] = xgb_bold_varios43
    print 'conjunto: {}, cv:{}'.format(i,np.mean(xgb_bold_varios45))              
                  

#### começã aqui o comentario
'''from random import shuffle
gb = train.columns
lista = [gb[0],gb[1],gb[2],gb[3],gb[4],gb[5],gb[6],gb[7],gb[8],gb[9],gb[10],gb[11], gb[12],gb[13]]  
exemplo = list(combinations(lista,5))
final = {}
shuffle(exemplo)
for i in exemplo:
   print(i) 
   nome = i[0] + '_' + i[1] + '_' + i[2] + '_' + i[3] + '_' + i[4]
   traintest1 = traintest.copy()
   traintest1[nome] = (traintest1[i[0]].map(str)+traintest1[i[1]].map(str)+traintest1[i[2]].map(str)+traintest1[i[3]].map(str)+traintest1[i[4]].map(str))
   #columnsToEncode = ['ps_ind_01_ps_ind_07_bin', 'ps_ind_01_ps_ind_03','ps_ind_01_ps_ind_15','ps_ind_01_ps_ind_08_bin','ps_ind_01_ps_ind_10_bin']
                      
   
   #columnsToEncode1 = [nome]
                       
   
   #gb10 = pd.value_counts(merda).keys()

  
   #for c in columnsToEncode:
       
   lbl = preprocessing.LabelEncoder()
   lbl.fit(list(traintest1[nome].values)) 
   traintest1[nome] = lbl.transform(list(traintest1[nome].values)) 
 

   #for c in columnsToEncode1:
    
   #qtd = pd.value_counts(traintest1[nome])
   #testando = lambda x: qtd[x]
   #traintest1[nome] = traintest1[nome].apply(testando) 


   train = traintest1.iloc[:train.shape[0],:]
   #teste = traintest1.iloc[train.shape[0]:,:]
   x_train = np.array(train)

   kfold = 5
   kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
   xgb_bold_varios = []
   for p, (train_index, test_index) in enumerate(kf.split(x_train, target_train)):
      train_X, valid_X = x_train[train_index], x_train[test_index]
      train_y, valid_y = target_train[train_index], target_train[test_index]

    

      d_train = xgb.DMatrix(train_X, train_y)
      d_valid = xgb.DMatrix(valid_X, valid_y)
    
      #d_test = xgb.DMatrix(teste)
    
      watchlist = [(d_train, 'train'), (d_valid, 'valid')]
      #clf = xgb.train(xgb_params_2, d_train, num_boost_round=5000,feval=gini_xgb, verbose_eval=50)    

      model = xgb.train(xgb_params_2, d_train, 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=False, early_stopping_rounds=100)
      xgb_bold_varios.append(model.best_score)  
      
   final[nome] = np.mean(xgb_bold_varios)   
   print 'conjunto: {}, cv:{}'.format(i,np.mean(xgb_bold_varios))          '''

# termina aqui o comentario    

#indices = np.arange(train.shape[0])
#train = np.array(train)
#teste = np.array(teste)
#train['ps_car_13'] = train.ps_car_13.apply(lambda x: round(x,6))
#teste['ps_car_13'] = teste.ps_car_13.apply(lambda x: round(x,6))
'''X_train, X_test, y_train, y_test = train_test_split(train, target_train, test_size=0.2, random_state=43)

d_train = xgb.DMatrix(X_train, y_train)
d_valid = xgb.DMatrix(X_test, y_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

model1_1_1 = xgb.train(xgb_params_2, d_train, 5000,  watchlist, maximize=True, verbose_eval=50,feval=gini_xgb, early_stopping_rounds=100)


X_train, X_test, y_train, y_test = train_test_split(train, target_train, test_size=0.2, random_state=44)

d_train = xgb.DMatrix(X_train, y_train)
d_valid = xgb.DMatrix(X_test, y_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

model1_1_2 = xgb.train(xgb_params_2, d_train, 5000,  watchlist, maximize=True, verbose_eval=50,feval=gini_xgb, early_stopping_rounds=100)


X_train, X_test, y_train, y_test = train_test_split(train, target_train, test_size=0.2, random_state=45)

d_train = xgb.DMatrix(X_train, y_train)
d_valid = xgb.DMatrix(X_test, y_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

model1_1_3 = xgb.train(xgb_params_2, d_train, 5000,  watchlist, maximize=True, verbose_eval=50,feval=gini_xgb, early_stopping_rounds=100)


X_train, X_test, y_train, y_test = train_test_split(train, target_train, test_size=0.2, random_state=46)

d_train = xgb.DMatrix(X_train, y_train)
d_valid = xgb.DMatrix(X_test, y_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

model1_1_4 = xgb.train(xgb_params_2, d_train, 5000,  watchlist, maximize=True, verbose_eval=50,feval=gini_xgb, early_stopping_rounds=100)


X_train, X_test, y_train, y_test = train_test_split(train, target_train, test_size=0.2, random_state=47)

d_train = xgb.DMatrix(X_train, y_train)
d_valid = xgb.DMatrix(X_test, y_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

model1_1_5 = xgb.train(xgb_params_2, d_train, 5000,  watchlist, maximize=True, verbose_eval=50,feval=gini_xgb, early_stopping_rounds=100)


X_train, X_test, y_train, y_test = train_test_split(train, target_train, test_size=0.2, random_state=48)

d_train = xgb.DMatrix(X_train, y_train)
d_valid = xgb.DMatrix(X_test, y_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

model1_1_6 = xgb.train(xgb_params_2, d_train, 5000,  watchlist, maximize=True, verbose_eval=50,feval=gini_xgb, early_stopping_rounds=100)


X_train, X_test, y_train, y_test = train_test_split(train, target_train, test_size=0.2, random_state=49)

d_train = xgb.DMatrix(X_train, y_train)
d_valid = xgb.DMatrix(X_test, y_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

model1_1_7 = xgb.train(xgb_params_2, d_train, 5000,  watchlist, maximize=True, verbose_eval=50,feval=gini_xgb, early_stopping_rounds=100)'''

###########################################################################

xgb_params_3 = {
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
    'monotone_constraints' : "(1,1,1,1,1,1,1,1,1,1,1)",
    'silent': True
     
                 }

#'monotone_constraints' : "(1,-1)",

X_train, X_test, y_train, y_test = train_test_split(train, target_train, test_size=0.2, random_state=42)

d_train = xgb.DMatrix(X_train, y_train)
d_valid = xgb.DMatrix(X_test, y_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

model2 = xgb.train(xgb_params_2, d_train, 5000,  watchlist, maximize=True, verbose_eval=50,feval=gini_xgb, early_stopping_rounds=100)
num_boost_rounds = model2.best_iteration
d_train_tudo = xgb.DMatrix(train, target_train)
clf = xgb.train(xgb_params_2, d_train_tudo, num_boost_round=2000,feval=gini_xgb, verbose_eval=10)

##############################################

d_test = xgb.DMatrix(teste)
xgb_pred = clf.predict(d_test)
output = pd.DataFrame({'id': id_test, 'target': xgb_pred})
output.to_csv('test-final1.csv', index=False)



#################################################################3


from random import shuffle
gb = traintest.columns
lista = [gb[0],gb[1],gb[2],gb[3],gb[4],gb[5],gb[6],gb[7],gb[8],gb[9],gb[10],gb[11], gb[12],gb[13]]  
exemplo = list(combinations(gb,2))
final = {}
shuffle(exemplo)
kfold = 5
kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
ntrain = train.shape[0]
for i in exemplo:
   #print(i) 
   nome = i[0] + '_' + i[1]
   #nome = i
   traintest1 = traintest.copy()
   traintest1[nome] = (traintest1[i[0]].map(str)+traintest1[i[1]].map(str))
   lbl = preprocessing.LabelEncoder()
   lbl.fit(list(traintest1[nome].values)) 
   traintest1[nome] = lbl.transform(list(traintest1[nome].values))                    
   
   train = traintest1.iloc[:train.shape[0],:]
   
   feature = funcoes.get_oof_encoding(train, target_train,ntrain, kf, nome)
   train['encoding'] = feature
   train['target'] = target_train
   print 'Feature: {}'.format(nome)
   #train.groupby("target")[nome].hist(bins=15,alpha=0.4)
   fig = plt.figure(figsize=(16,6))
   ax0 = fig.add_subplot(121)
   #train.groupby("target")[nome].plot(kind='kde')
   train.groupby("target")[nome].hist(bins='auto',alpha=0.4)
   #plt.savefig('fig/'+nome+'.png')
   #plt.show()
   ax1 = fig.add_subplot(122)
 
   #train.groupby("target")['encoding'].plot(kind='kde')
   train.groupby("target")['encoding'].hist(bins='auto',alpha=0.4)
   #plt.savefig('fig/'+nome+'-encoding'+'.png')
   #plt.show()
   plt.tight_layout()
   plt.savefig('figs2/'+nome+'.png')
   #teste = traintest1.iloc[train.shape[0]:,:]
   


exemplo = list(combinations(gb,4))
final = {}
shuffle(exemplo)
kfold = 5
kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
ntrain = train.shape[0]
for i in exemplo:
   print(i) 
   nome = i[0] + 'X' + i[1] + 'X' + i[2] + 'X' + i[3]
   #print 'Feature: {}'.format(nome)
   #nome = i
   traintest1 = traintest.copy()
   traintest1[nome] = (traintest1[i[0]].map(str)+traintest1[i[1]].map(str)+traintest1[i[2]].map(str)+traintest1[i[3]].map(str))
   lbl = preprocessing.LabelEncoder()
   lbl.fit(list(traintest1[nome].values)) 
   traintest1[nome] = lbl.transform(list(traintest1[nome].values))                    
   
   train = traintest1.iloc[:train.shape[0],:]
   
   feature = funcoes.get_oof_encoding(train, target_train,ntrain, kf, nome)
   train['encoding'] = feature
   train['target'] = target_train
   #print 'Feature: {}'.format(nome)
   #train.groupby("target")[nome].hist(bins=15,alpha=0.4)
   fig = plt.figure(figsize=(16,6))
   ax0 = fig.add_subplot(121)
   #train.groupby("target")[nome].plot(kind='kde')
   train.groupby("target")[nome].hist(bins='auto',alpha=0.4,ylim=(0,50000))
   #plt.savefig('fig/'+nome+'.png')
   #plt.show()
   ax1 = fig.add_subplot(122)
 
   #train.groupby("target")['encoding'].plot(kind='kde')
   train.groupby("target")['encoding'].hist(bins='auto',alpha=0.4, ylim=(0,50000))
   #plt.savefig('fig/'+nome+'-encoding'+'.png')
   #plt.show()
   plt.tight_layout()
   plt.savefig('figs4/'+nome+'.png')



#iris.groupby("Name").PetalWidth.plot(kind='kde', ax=axs[1])

'''for j in train.columns:
 name_1 = j
 for i in train.columns:
    fig = plt.figure(figsize=(8,8))
    plt.scatter(train[name_1].fillna(-10), train[i].fillna(-10), c=train.target)
    plt.ylabel(i)
    plt.xlabel(name_1)
    caminho = 'fig/' + name_1 + '_' + i + '.png'
    plt.savefig(caminho)'''

'''gb1 = train[train.ps_car_13 == 0.674583]
gb = gb1.copy()
alvo = gb.target
gb.drop('target', axis=1, inplace=True)

pd.scatter_matrix(gb.iloc[:,0:5].fillna(-10), c=alvo, figsize=(15, 15), marker='o',
                  hist_kwds={'bins': 20}, s=60, alpha=.8)

plt.savefig('foo.png')

number_0 = gb1[gb1.target == 0]
number_1 = gb1[gb1.target == 1]'''

#colors = ['red', 'blue'] 







from sklearn.cluster import KMeans  
kmeans = KMeans(n_clusters=2, random_state=0).fit(gb.fillna(-10))

#res = principal.iniciar(train,target_train,idx1, idx2)


'''kfold = 5
kf = StratifiedKFold(n_splits=kfold, random_state=42)

for i, (train_index, test_index) in enumerate(kf.split(train, target_train)):
    train_X, valid_X = train[train_index], train[test_index]
    train_y, valid_y = target_train[train_index], target_train[test_index]

    # params configuration also from the1owl's kernel
    # https://www.kaggle.com/the1owl/forza-baseline
    #xgb_params = {'eta': 0.02, 'max_depth': 5, 'subsample': 0.9, 'colsample_bytree': 0.9, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

    d_train = xgb.DMatrix(train_X, train_y)
    d_valid = xgb.DMatrix(valid_X, valid_y)
    d_test = xgb.DMatrix(teste)
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    #clf = xgb.train(xgb_params_2, d_train, num_boost_round=5000,feval=gini_xgb, verbose_eval=50)    

    model = xgb.train(xgb_params_2, d_train, 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=100)
                        
    xgb_pred = model.predict(d_test)
    xgb_preds.append(list(xgb_pred))
    
    
    
preds=[]
for i in range(len(xgb_preds[0])):
    sum=0
    for j in range(kfold):
        sum+=xgb_preds[j][i]
    preds.append(sum / kfold)

output = pd.DataFrame({'id': id_test, 'target': preds})
output.to_csv("{}-foldCV_avg_sub.csv".format(kfold), index=False)

##########################################################################

d_test = xgb.DMatrix(teste)
xgb_pred = clf.predict(d_test)
output = pd.DataFrame({'id': id_test, 'target': xgb_pred})
output.to_csv("xgb_param2.csv", index=False)

sel = [7]
ind = train[train.iloc[:,0].isin(sel)].index
treino_1 = train.loc[ind]
print(pd.value_counts(treino_1.target))'''
    
#############################################  



'''sel = [1, 3, 6]
ind = train[train.iloc[:,0].isin(sel)].index
treino_1 = train.loc[ind]
y_trein_1 = treino_1.target
treino_1 = treino_1.drop('target', axis=1)
    
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(treino_1, y_trein_1, test_size=0.2, random_state=42)

d_train_1 = xgb.DMatrix(X_train_1, y_train_1)
d_valid_1 = xgb.DMatrix(X_test_1, y_test_1)
watchlist_1 = [(d_train_1, 'train'), (d_valid_1, 'valid')]

model_train_1 = xgb.train(xgb_params_2, d_train_1, 5000,  watchlist_1, maximize=True, verbose_eval=False,feval=gini_xgb, early_stopping_rounds=100)
   
num_boost_rounds = model_train_1.best_iteration
d_train_tudo = xgb.DMatrix(treino_1, y_trein_1)

clf = xgb.train(xgb_params_2, d_train_tudo, num_boost_round=num_boost_rounds,feval=gini_xgb, verbose_eval=10)

####################################################################

sel = [0, 2, 5, 4]
ind = train[train.iloc[:,0].isin(sel)].index
treino_1 = train.loc[ind]
y_trein_1 = treino_1.target
treino_1 = treino_1.drop('target', axis=1)
    
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(treino_1, y_trein_1, test_size=0.2, random_state=42)

d_train_1 = xgb.DMatrix(X_train_1, y_train_1)
d_valid_1 = xgb.DMatrix(X_test_1, y_test_1)
watchlist_1 = [(d_train_1, 'train'), (d_valid_1, 'valid')]

model_train_2 = xgb.train(xgb_params_2, d_train_1, 5000,  watchlist_1, maximize=True, verbose_eval=False,feval=gini_xgb, early_stopping_rounds=100)
   
num_boost_rounds = model_train_2.best_iteration


d_train_tudo = xgb.DMatrix(treino_1, y_trein_1)

clf_1 = xgb.train(xgb_params_2, d_train_tudo, num_boost_round=num_boost_rounds,feval=gini_xgb, verbose_eval=10) 

###################################################################

sel = [7]
ind = train[train.iloc[:,0].isin(sel)].index
treino_1 = train.loc[ind]
y_trein_1 = treino_1.target
treino_1 = treino_1.drop('target', axis=1)
    
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(treino_1, y_trein_1, test_size=0.2, random_state=42)

d_train_1 = xgb.DMatrix(X_train_1, y_train_1)
d_valid_1 = xgb.DMatrix(X_test_1, y_test_1)
watchlist_1 = [(d_train_1, 'train'), (d_valid_1, 'valid')]

model_train_3 = xgb.train(xgb_params_2, d_train_1, 5000,  watchlist_1, maximize=True, verbose_eval=False,feval=gini_xgb, early_stopping_rounds=100)
num_boost_rounds = model_train_3.best_iteration

d_train_tudo = xgb.DMatrix(treino_1, y_trein_1)

clf_2 = xgb.train(xgb_params_2, d_train_tudo, num_boost_round=num_boost_rounds,feval=gini_xgb, verbose_eval=10) '''


##########################################3

#teste['id'] = id_test

'''sel = [1, 3, 6]
ind = teste[teste.iloc[:,0].isin(sel)].index
teste_1 = teste.loc[ind]
id_1 = teste_1.id
teste_1=teste_1.drop(['id'], axis = 1)
d_test = xgb.DMatrix(teste_1)
xgb_pred_1 = clf.predict(d_test)



sel = [0, 2, 5, 4]
ind = teste[teste.iloc[:,0].isin(sel)].index
teste_2 = teste.loc[ind]
id_2 = teste_2.id
teste_2=teste_2.drop(['id'], axis = 1)
d_test = xgb.DMatrix(teste_2)
xgb_pred_2 = clf_1.predict(d_test)


sel = [7]
ind = teste[teste.iloc[:,0].isin(sel)].index
teste_3 = teste.loc[ind]
id_3 = teste_3.id
teste_3=teste_3.drop(['id'], axis = 1)
d_test = xgb.DMatrix(teste_3)
xgb_pred_3 = clf_2.predict(d_test)


output_1 = pd.DataFrame({'id': id_1, 'target': xgb_pred_1})
output_2 = pd.DataFrame({'id': id_2, 'target': xgb_pred_2})
output_3 = pd.DataFrame({'id': id_3, 'target': xgb_pred_3})


out_final = pd.concat([output_1,output_2,output_3], axis=0)



out_final.to_csv("xgb_param10.csv", index=False)'''

'''from sklearn.ensemble import ExtraTreesClassifier
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
   }
et = ensemble.SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
xgbo_1_oof_train = funcoes.get_oof(et, train, target_train,ntrain,kf) '''

import funcoes as funcoes
reload(funcoes)                
import Ensemble as ensemble
reload(ensemble)
train = np.array(train)
teste = np.array(teste)
xgb_params_2 = {
    'eta': 0.02,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'silent': True
     
                 }
                
ntrain = train.shape[0]
SEED = 42
NFOLDS = 5 
#kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
kf = StratifiedKFold(n_splits=NFOLDS, random_state=SEED)

xgbo_1 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_2,flag=1)                
xgbo_1_oof_train = funcoes.get_oof(xgbo_1, train, target_train,ntrain,kf) 

x_train_inter = np.concatenate((xgbo_1_oof_train, xgbo_2_oof_train, xgbo_3_oof_train_1,xgbo_4_oof_train,xgbo_5_oof_train,xgbo_6_oof_train), axis=1)
                
  
treino = np.array(train)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)

all_objects = []

for i in range(0,train.shape[0]):
    print 'Indice: {}'.format(i)
    all_objects.append(neigh.kneighbors(train.iloc[i,:].fillna(-999).values.reshape(1,-1), return_distance=False)[0])  


perce_0 = []
perce_1 = []

for i in range(0,len(all_objects)):
    print 'Indice: {}'.format(i)
    val = pd.value_counts(target_train[all_objects[i]])
    if 1 in val.keys():
       perce = val[1] / float(500)  
       perce_1.append(perce)
       perce_0.append(1 - perce)
       
    else:
       perce_0.append(1)
       perce_1.append(0)
        
############################333    
all_objects_teste = []

for i in range(0,teste.shape[0]):
    print 'Indice: {}'.format(i)
    all_objects_teste.append(neigh.kneighbors(teste.iloc[i,:].fillna(-999).values.reshape(1,-1), return_distance=False)[0])  





perce_0_teste = []
perce_1_teste = []

for i in range(0,len(all_objects_teste)):
    print 'Indice: {}'.format(i)
    val = pd.value_counts(target_train[all_objects_teste[i]])
    if 1 in val.keys():
       perce = val[1] / float(500)  
       perce_1_teste.append(perce)
       perce_0_teste.append(1 - perce)
       
    else:
       perce_0.append(1)
       perce_1.append(0)


from sklearn.preprocessing import MinMaxScaler
legal = np.where(target_train == 1)[0]
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train.fillna(-999))
neigh = KNeighborsClassifier(n_neighbors=3, n_jobs=2)
neigh.fit(scaler.transform(train.fillna(-999)), target_train)
train_scaled = scaler.transform(train.fillna(-999))

   
tudo = []   
for i in range(0,21694,200):
      print(i)
      gb = target_train[neigh.kneighbors(train_scaled[legal[i:i+200],:], return_distance=False)]
      tudo.append(gb)
      #zero.append(len(np.where(gb == 0)[0]))
      #um.append(len(np.where(gb == 1)[0]))
zero = []
um = []   
for k in tudo:
       for m in k:
          zero.append(len(np.where(m == 0)[0])) 
          um.append(len(np.where(m == 1)[0]))

   
   

traintest['merda'] = (traintest['ps_calc_10'].map(str)+traintest['ps_calc_11'].map(str)+traintest['ps_calc_12'].map(str))
train = traintest.iloc[:train.shape[0],:]
gb = pd.value_counts(traintest.merda).keys()
ind = train[train.merda == gb[6]].index   
gb1 = pd.value_counts(target_train[ind])

print 'aqui:{}'.format(gb1[1] / float(gb1[1] + gb1[0]))
           
legal = ['first', 'ps_ind_03_freq', 'freq_01','freq_05', 'freq_06', 'ps_ind_01_ps_ind_07_bin', 'ps_ind_01_ps_ind_03','ps_ind_01_ps_ind_15'
         ,'ps_ind_01_ps_ind_08_bin', 'ps_ind_01_ps_ind_10_bin']
colunas = train.columns

for i in legal[2:3]:
   print 'coluna:{}'.format(i)
   gb = pd.value_counts(train[i]).keys()
   for j in gb:
      #print 'coluna_indice:{}'.format(j)
      ind = train[train[i] == j].index   
      gb1 = pd.value_counts(target_train[ind])

      print 'coluna_indice:{}, aqui:{}'.format(j,gb1[1] / float(gb1[1] + gb1[0]))   
           
   print('####################################')         







from itertools import combinations
gb = train.columns
lista = [gb[0],gb[1],gb[2],gb[3],gb[4],gb[5],gb[6],gb[7],gb[8],gb[9],gb[10],gb[11], gb[12],gb[13]]  
exemplo = list(combinations(lista,5))

for i in exemplo:
     
   merda = (train[i[0]].map(str)+train[i[1]].map(str)+train[i[2]].map(str)+train[i[3]].map(str))
   gb10 = pd.value_counts(merda).keys()
   '''for j in gb10:
       ind = merda[merda == j].index   
       gb20 = pd.value_counts(target_train[ind])

       print 'coluna_indice:{}, aqui:{}'.format(j,gb20[1] / float(gb20[1] + gb20[0]))  ''' 
          
shuffle(exemplo)
       
i = exemplo[1]       
merda = (train[i[0]].map(str)+train[i[1]].map(str)+train[i[2]].map(str)+train[i[3]].map(str)+train[i[4]].map(str))       
gb = pd.value_counts(merda).keys()
ind = merda[merda == '1.00.0000'].index   
gb1 = pd.value_counts(target_train[ind])
 
teste = traintest.iloc[train.shape[0]:,:]
tam = teste[teste[legal[0]] == 311]       




for i in range(200,1000,10):
    print(pd.value_counts(np.diff(ind)[0:i]))


########################################################################################


import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

# Regularized Greedy Forest


class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        #results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        #print("Stacker score: %.5f" % (results.mean()))
        
        
        random_state = [42]
        xgb_bold_final2 = {}

        for k in random_state:
        print 'Random state: {}'.format(k)  
        xgb_bold_varios52 = []
  
  
        kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=k)   
        for p, (train_index, test_index) in enumerate(kf.split(x_train, target_train)):
           train_X, valid_X = x_train.iloc[train_index], x_train.iloc[test_index]
           train_y, valid_y = target_train[train_index], target_train[test_index]
      
      

      

          d_train = xgb.DMatrix(train_X, train_y)
          d_valid = xgb.DMatrix(valid_X, valid_y)
    
    
          watchlist = [(d_train, 'train'), (d_valid, 'valid')]

          model = xgb.train(xgb_params_2, d_train, 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=100)
          xgb_bold_varios52.append(model.best_score)  
     
                 
                 
                 
        print 'Media:{}'.format(np.mean(xgb_bold_varios52))         
        X_train, X_test, y_train, y_test = train_test_split(S_train, y, test_size=0.2, random_state=42)

        d_train = xgb.DMatrix(X_train, y_train)
        d_valid = xgb.DMatrix(X_test, y_test)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        model2 = xgb.train(xgb_params_2, d_train, 5000,  watchlist, maximize=True, verbose_eval=50,feval=gini_xgb, early_stopping_rounds=100)
        num_boost_rounds = model2.best_iteration
        d_train_tudo = xgb.DMatrix(train, target_train)
        clf1 = xgb.train(xgb_params_2, d_train_tudo, num_boost_round=num_boost_rounds,feval=gini_xgb, verbose_eval=10)
       
##############################################



        d_test = xgb.DMatrix(T)
        xgb_pred = clf.predict(d_test)
        #self.stacker.fit(S_train, y)
        #res = clf1.predict_proba(S_test)[:,1]
        return xgb_pred


        
# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 500
lgb_params['seed'] = 99


lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['seed'] = 99


lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['seed'] = 99





lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)

#rf_model = RandomForestClassifier(**rf_params)

#et_model = ExtraTreesClassifier(**et_params)
        
xgb_model = XGBClassifier(**xgb_params_2)

#cat_model = CatBoostClassifier(**cat_params)

#rgf_model = RGFClassifier(**rgf_params) 

#gb_model = GradientBoostingClassifier(max_depth=5)

#ada_model = AdaBoostClassifier()

#log_model = LogisticRegression()

log_model = XGBClassifier(**xgb_params_2)
        
stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (lgb_model, lgb_model2, lgb_model3,xgb_model))        
        
y_pred = stack.fit_predict(train, target_train, teste)        



sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('stacked_1.csv', index=False)




