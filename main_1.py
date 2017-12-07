# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 19:20:54 2017

@author: rebli
"""
'''import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev0\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']'''

from sklearn.cross_validation import KFold;
          
import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from sklearn import model_selection, preprocessing
import Principal as principal
reload(principal)
import lightgbm as lgb
import funcoes as funcoes
reload(funcoes)
import Ensemble as ensemble
reload(ensemble)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error    
        
        

train = pd.read_csv('train_2016_v2.csv',parse_dates=["transactiondate"])
prop = pd.read_csv('properties_2016.csv')
sample = pd.read_csv('sample_submission.csv')

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)



ulimit = np.percentile(train.logerror.values, 98)
llimit = np.percentile(train.logerror.values, 2)

superior = train[train.logerror > ulimit].index
inferior = train[train.logerror < llimit].index

train = train.drop(superior, axis=0)
train = train.drop(inferior, axis=0)            
#train['logerror'].ix[train['logerror']>ulimit] = ulimit
#train['logerror'].ix[train['logerror']<llimit] = llimit

     








train['transaction_month'] = train['transactiondate'].dt.month

train['transaction_week'] = train['transactiondate'].dt.week

#ind, = np.where(train.transaction_month == 12)     

#plt.savefig("fig/figura1.png")     

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

#excluidos 'finishedsquarefeet13'

x_train = df_train.drop(['parcelid', 'transactiondate','propertycountylandusecode','propertyzoningdesc','assessmentyear'], axis=1)
#y_train = df_train['logerror'].values
                    
 
                              
#train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)
    
    
y_train = x_train['logerror'].values

x_train['area_finishedfloor1squarefeet_bedroomcnt'] = x_train['finishedfloor1squarefeet'] / x_train['bedroomcnt']
x_train['finishedsquarefeet15_regionidcity'] = x_train['regionidcity'] / x_train['finishedsquarefeet15']

x_train.drop(['logerror','transaction_month','transaction_week'], axis=1, inplace=True)

###########
#x_train['censustractandblock_heatingorsystemtypeid'] = x_train['heatingorsystemtypeid'] / x_train['censustractandblock']


'''x_train['finishedsquarefeet12_lotsizesquarefeet'] = x_train['lotsizesquarefeet'] - x_train['finishedsquarefeet12']

x_train['censustractandblock_propertylandusetypeid'] = x_train['propertylandusetypeid'] / x_train['censustractandblock']
x_train['structuretaxvaluedollarcnt_landtaxvaluedollarcnt'] = x_train['landtaxvaluedollarcnt'] / x_train['structuretaxvaluedollarcnt']
x_train['bedroomcnt'] = np.log(x_train['bedroomcnt'])
x_train['censustractandblock'] = np.log(x_train['censustractandblock'])

x_train['D_Prop_Age'] = 2018 - x_train['yearbuilt']   
#Total number of rooms
x_train['D_TotalRooms'] = x_train['bathroomcnt']*x_train['bedroomcnt']
#Missing Count
x_train['D_miss_count']=x_train.apply(lambda x: sum(x.isnull().values), axis = 1)
x_train['assessmentyear'].value_counts()

x_train['calculatedfinishedsquarefeet_yearbuilt'] = x_train['yearbuilt'] / x_train['calculatedfinishedsquarefeet']
x_train['bedroomcnt_taxdelinquencyyear'] = x_train['taxdelinquencyyear'] / x_train['bedroomcnt']
x_train['calculatedfinishedsquarefeet_lotsizesquarefeet'] = x_train['lotsizesquarefeet'] / x_train['calculatedfinishedsquarefeet']
x_train['buildingqualitytypeid_yearbuilt'] = x_train['yearbuilt'] / x_train['buildingqualitytypeid']
x_train['finishedsquarefeet12_regionidzip'] = x_train['regionidzip'] / x_train['finishedsquarefeet12']
x_train['garagecarcnt_buildingqualitytypeid'] = x_train['buildingqualitytypeid'] / x_train['garagecarcnt']

x_train['primeiro'] = x_train['lotsizesquarefeet'] - x_train['finishedsquarefeet12']'''

###################################################################

xgb_params = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 0
     
                 }


index_treino, index_validacao = funcoes.cria_treino_validacao(x_train)

treino = x_train.iloc[index_treino]
validacao = x_train.iloc[index_validacao]

y_treino = treino.logerror
treino = treino.drop(['logerror','transaction_month','transaction_week'], axis=1)
y_validacao = validacao.logerror
validacao = validacao.drop(['logerror','transaction_month','transaction_week'], axis=1)



'''y_train = x_train.logerror
x_train = x_train.drop(['logerror','transaction_month','transaction_week'], axis=1)

result = principal.iniciar(x_train, y_train,index_treino,index_validacao)'''


'''y_treino = y_train[index_treino]
y_validacao = y_train[index_validacao]
treino = treino.iloc[:,lista]
validacao = validacao.iloc[:,lista]'''

treino_columns = treino.columns
d_train = xgb.DMatrix(treino, label=y_treino)
d_validacao = xgb.DMatrix(validacao, label=y_validacao)

cross_validation = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100,
                           verbose_eval=50,show_stdv=False)
num_boost_rounds = len(cross_validation)
watchlist = [(d_train, 'train'), (d_validacao, 'valid')]
clf = xgb.train(xgb_params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=50)


#######################################################################################3


x_train = x_train.drop(['logerror','transaction_month','transaction_week'], axis=1)


'''x_train['bins_lotsizesquarefeet'] = pd.cut(x_train['lotsizesquarefeet'], np.percentile(x_train['lotsizesquarefeet'].dropna(), [2.5,5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30,32.5,35,37.5,40,42.5,45,47.5,50,52.5,55,57.5,60,62.5,65,67.5,70,72.5,75,77.5,80,82.5,85,87.5,90,92.5,95,97.5,100]))
#x_train['bins_lotsizesquarefeet1'] = pd.cut(x_train['lotsizesquarefeet'], np.percentile(x_train['lotsizesquarefeet'].dropna(), [10,20,30,40,50,60,70,80,90,100]))
#x_train['bins_lotsizesquarefeet2'] = pd.cut(x_train['lotsizesquarefeet'], np.percentile(x_train['lotsizesquarefeet'].dropna(), [5,15,20,40,60,80,100]))

columnsToEncode = ['bins_lotsizesquarefeet']   
for c in columnsToEncode:
    
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))'''
        
        
      
                

'''minimo = min(x_train.yearbuilt)
contando = lambda x: x - minimo
x_train['contando'] = x_train['yearbuilt'].apply(contando)'''
#d_validacao = xgb.DMatrix(validacao, label=y_validacao)
#d_valid = xgb.DMatrix(x_valid, label=y_valid)


'''lista = ['calculatedfinishedsquarefeet','finishedsquarefeet6',
         'finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15','finishedsquarefeet50']'''

            
lista = ['lotsizesquarefeet','finishedfloor1squarefeet', 'calculatedfinishedsquarefeet','finishedsquarefeet6',
           'finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15','finishedsquarefeet50',
           'garagetotalsqft']
divisor =  'fullbathcnt'

         
#divisor = 'fullbathcnt' #array([ 0.046895])
#divisor = 'bathroomcnt' #array([ 0.04690767])
#divisor = 'bedroomcnt' #array([ 0.04689667])
#divisor = 'calculatedbathnbr' # array([ 0.04690267])
#divisor = 'threequarterbathnbr' # array([ 0.046903])

xgb_params = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 0
     
                 }
colunas = treino.columns
contando = 0
lk10_roomcnt = {}
for j in colunas:
    print "testando:{}, numero:{}".format(j,contando)
    contando += 1
    #x_train1 = x_train.copy()
    #x_train1[j] = np.log(x_train1[j])
    for k in colunas1:
       nome = 'primeiro'
       x_train[nome] = x_train[k] / x_train[j]
       treino = x_train.iloc[index_treino]
       validacao = x_train.iloc[index_validacao]

       y_treino = treino.logerror
       treino = treino.drop(['logerror','transaction_month','transaction_week'], axis=1)
       y_validacao = validacao.logerror
       validacao = validacao.drop(['logerror','transaction_month','transaction_week'], axis=1)
       #x_train[nome] = x_train[lista] / x_train[j]
       #x_train2 = x_train1.copy()
       #x_train2[k] = np.log(x_train2[k])
       d_train = xgb.DMatrix(treino, label=y_treino)
       d_validacao = xgb.DMatrix(validacao, label=y_validacao)
       watchlist = [(d_train, 'train'), (d_validacao, 'valid')]
       clf = xgb.train(xgb_params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=False)
       '''cross_validation = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100,
                           verbose_eval=50,show_stdv=False)'''
       nome1 = j + "_" + k
       lk10_roomcnt[nome1] = np.array(clf.best_score)
    
    print "melhor ate agora:{}".format(lk10_roomcnt[min(lk10_roomcnt, key=lk10_roomcnt.get)])
#del x_train; gc.collect()

#print('Training ...')
 
'''from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0, perplexity=50, n_iter=2000)
np.set_printoptions(suppress=True)
dados = model.fit_transform(x_train[vb]) '''

num = 4
colun = x_train.columns
print(x_train[colun[num]].isnull().sum())
plt.plot(x_train[colun[num]], 'o')
plt.show()
pd.value_counts(x_train[colun[num]])

plt.hist(x_train[colun[num]], bins=20)

#d_train = xgb.DMatrix(x_train, label=y_train)

#Informações importantes
# features: bathroomcnt,

xgb_params = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 0
     
                 }

#y_predict = np.exp(y_predict) - 1
#x_train = x_train.drop('primeiro', axis=1)

train_columns = x_train.columns
d_train = xgb.DMatrix(x_train, label=y_train)

cross_validation = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100,
                           verbose_eval=50,show_stdv=False)

num_boost_rounds = len(cross_validation)
clf = xgb.train(xgb_params, d_train, num_boost_round=num_boost_rounds, verbose_eval=10)


d_train = xgb.DMatrix(x_train)
pred = clf.predict(d_train)
x_train['first_pred'] = np.log1p(pred)
x_train['first_pred1'] = np.log(pred)
x_train['first_pred2'] = np.sqrt(pred)
x_train['first_pred3'] = pred


d_train = xgb.DMatrix(x_train, label=y_train)
cross_validation = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100,
                           verbose_eval=50,show_stdv=False)

num_boost_rounds = len(cross_validation)       
clf1 = xgb.train(xgb_params, d_train, num_boost_round=num_boost_rounds, verbose_eval=10)


'''d_train1 = xgb.DMatrix(x_train)
pred = clf.predict(d_train1)

x_train['residuo'] = y_train - pred
plt.figure(figsize=(12,8))
sns.boxplot(x="regionidcounty", y="residuo", data=x_train)
plt.ylabel('residuo', fontsize=12)
plt.xlabel('Bathroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("How log error changes with bathroom count?", fontsize=15)
plt.show()'''

###################################################################
train_columns = x_train.columns

lgb_train = lgb.Dataset(x_train, y_train)
#lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict


      
      
params = {
    'max_bin': 10,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'sub_feature': 0.5,
    'num_leaves': 512,
    'learning_rate': 0.0021,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.85,
    'bagging_freq': 40,
    'min_data': 500,
    'min_hessian': 0.05,
    'verbose': 0
}

print('Start training...')

cross_validation = lgb.cv(params, lgb_train, num_boost_round=1000, early_stopping_rounds=100,
                           verbose_eval=50,show_stdv=False,nfold=3 )

num_boost_rounds = len(cross_validation['l1-mean'])
gbm = lgb.train(params,
                lgb_train,430,verbose_eval=10)






################################################

'''d_train1 = xgb.DMatrix(x_train)
pred = clf.predict(d_train1)
from sklearn.metrics import mean_absolute_error
plt.plot(pred[0:1000],y_train[0:1000], 'o')
plt.xlabel("pred")
plt.ylabel("verdadeiro")

from sklearn.metrics import r2_score

diff = y_train - pred
plt.plot(diff, 'o')
binwidth = 10
plt.hist(diff, bins=20)'''
##############################################################


'''xgb_params2 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 0,
     'seed': 20
                 }

cv_output2 = xgb.cv(xgb_params2, d_train, num_boost_round=1000, early_stopping_rounds=100,
    verbose_eval=50,show_stdv=False, seed=30)

num_boost_rounds2 = len(cv_output2)
#watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#model1 = xgb.train(xgb_params, d_train, num_boost_round=10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

clf2 = xgb.train(xgb_params2, d_train, num_boost_round=num_boost_rounds2, verbose_eval=10)'''


#####################################################################
xgb_params3 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 8,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 0
     
                 }

'''xgb_params3 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 0,
    'seed' : 15
     
                 }'''

cv_output3 = xgb.cv(xgb_params3, d_train, num_boost_round=1000, early_stopping_rounds=100,
    verbose_eval=50,show_stdv=False, seed=25)

num_boost_rounds3 = len(cv_output3)
#watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#model1 = xgb.train(xgb_params, d_train, num_boost_round=10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

clf3 = xgb.train(xgb_params3, d_train, num_boost_round=num_boost_rounds3, verbose_eval=10)

##########################################################################

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')

#####
df_test['area_finishedfloor1squarefeet_bedroomcnt'] = df_test['finishedfloor1squarefeet'] / df_test['bedroomcnt']
#df_test['finishedsquarefeet12_lotsizesquarefeet'] = df_test['lotsizesquarefeet'] - df_test['finishedsquarefeet12']
df_test['finishedsquarefeet15_regionidcity'] = df_test['regionidcity'] / df_test['finishedsquarefeet15']

########
#df_test['censustractandblock_heatingorsystemtypeid'] = df_test['heatingorsystemtypeid'] / df_test['censustractandblock']


'''df_test['finishedsquarefeet12_lotsizesquarefeet'] = df_test['lotsizesquarefeet'] - df_test['finishedsquarefeet12']

df_test['censustractandblock_propertylandusetypeid'] = df_test['propertylandusetypeid'] / df_test['censustractandblock']
df_test['structuretaxvaluedollarcnt_landtaxvaluedollarcnt'] = df_test['landtaxvaluedollarcnt'] / df_test['structuretaxvaluedollarcnt']
df_test['bedroomcnt'] = np.log(df_test['bedroomcnt'])
df_test['censustractandblock'] = np.log(df_test['censustractandblock'])

df_test['D_Prop_Age'] = 2018 - df_test['yearbuilt']   
#Total number of rooms
df_test['D_TotalRooms'] = df_test['bathroomcnt']*df_test['bedroomcnt']
#Missing Count
df_test['D_miss_count']=df_test.apply(lambda x: sum(x.isnull().values), axis = 1)
df_test['assessmentyear'].value_counts()

df_test['calculatedfinishedsquarefeet_yearbuilt'] = df_test['yearbuilt'] / df_test['calculatedfinishedsquarefeet']
df_test['bedroomcnt_taxdelinquencyyear'] = df_test['taxdelinquencyyear'] / df_test['bedroomcnt']
df_test['calculatedfinishedsquarefeet_lotsizesquarefeet'] = df_test['lotsizesquarefeet'] / df_test['calculatedfinishedsquarefeet']
df_test['buildingqualitytypeid_yearbuilt'] = df_test['yearbuilt'] / df_test['buildingqualitytypeid']
df_test['finishedsquarefeet12_regionidzip'] = df_test['regionidzip'] / df_test['finishedsquarefeet12']
df_test['garagecarcnt_buildingqualitytypeid'] = df_test['buildingqualitytypeid'] / df_test['garagecarcnt']

df_test['primeiro'] = df_test['lotsizesquarefeet'] - df_test['finishedsquarefeet12']'''





#del prop; gc.collect()


x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

#del df_test, sample; gc.collect()

d_test = xgb.DMatrix(x_test)
#lgb_test = lgb.Dataset(x_test)

#del x_test; gc.collect()

print('Predicting on test ...')

p_test = clf.predict(d_test)
x_test['first_pred'] = np.log1p(p_test)
x_test['first_pred1'] = np.log(p_test)
x_test['first_pred2'] = np.sqrt(p_test)
x_test['first_pred3'] = p_test

d_test = xgb.DMatrix(x_test)
p_test = clf1.predict(d_test)

#p_test_lgb = gbm.predict(x_test)
#p_test2 = clf2.predict(d_test)
#p_test3 = clf3.predict(d_test)



#del d_test; gc.collect()

sub = pd.read_csv('sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test
       
print('Writing csv ...')
sub.to_csv('xgb_starter.csv', index=False, float_format='%.4f') 





def excluindo_valores(x_train,y_train, colunas):
    lista = {}
    xgb_params = {
          'eta': 0.05,
          #'learning_rate': 0.05,
          'max_depth': 5,
          'subsample': 0.85,
          'colsample_bytree': 0.75,
          'min_child_weight': 7,
          'objective': 'reg:linear',
          'eval_metric': 'mae',
          'silent': 0
     
                 }
    
    
    for i in colunas:
       print "Excluindo....{}".format(i) 
       dados = x_train.drop(i,axis=1)
       
       d_train = xgb.DMatrix(dados, label=y_train)

       cv_output = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100,
            verbose_eval=50,show_stdv=False)
    
       lista[i] = round(cv_output[-1:]['test-mae-mean'],6)
    
    return lista


#############################################################3

'''anos = pd.unique(x_train.yearbuilt)

for i in anos:
    base = x_train[x_train.yearbuilt == anos[0]]
    #base.logerror.plot(kind='kde')
    base.boxplot(column='logerror')
    plt.show()'''


from matplotlib import pyplot
xgb.plot_importance(clf, max_num_features=20, height=0.6)
pyplot.show()

d_train1 = xgb.DMatrix(x_train)
pred = clf.predict(d_train1)
from sklearn.metrics import mean_absolute_error
plt.plot(pred[0:1000],y_train[0:1000], 'o')
plt.xlabel("pred")
plt.ylabel("verdadeiro")


residuo = y_train - pred

plt.plot(y_train,residuo, 'o')
plt.xlabel('verdadeiro')
plt.ylabel('residuo')

xgb_params = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1
     
                 }
  
  
#####################################################################3
x_train1 = x_train.copy()
#x_train1['pred'] = pred
x_train1['verd'] = y_train
#x_train1['diff1'] = (y_train - pred)        
cols = pd.value_counts(x_train1.regionidcity).keys()

lista = []
import itertools
poss = list(itertools.combinations(cols,3))


tudo_final = []


for i in range(0,1):
  res_final_7_final_1rodada = {}
  res_final_7_temp_1rodada = {}

  #verdadeiros = []
  #preditos = []
  modelos = {}
  vazio = []
  pontuacao = 1000000000000000000000000000000000000000000000
  for i in range(0,10):
    
    vazio = np.random.choice(cols,10, replace=False)
    vazio = list(vazio)
    preditos = []
    verdadeiros = []

    preditos_1 = []
    verdadeiros_1 = []
  
    #vazio.append(i)
    if 'NaN' not in vazio:
        ind = x_train1[x_train1.regionidcity.isin(vazio)].index
        teste = x_train1.loc[ind]
        res = x_train1.loc[list(set(x_train1.index) - set(ind))]
    if 'NaN' in vazio :
        index1 = x_train1.regionidcity.index[x_train1.regionidcity.apply(np.isnan)]
        ind = x_train1[x_train1.regionidcity.isin(vazio)].index
        x_train_interm = x_train1.loc[index1]
        teste = x_train1.loc[ind]
        teste = pd.concat([x_train_interm, teste] , axis=0)
        tudo = np.concatenate([index1,ind])
        
        res = x_train1.loc[list(set(x_train1.index) - set(tudo))]
        
        
    y_train_teste = teste.verd
    teste = teste.drop('verd', axis=1)
    
    y_train_resto = res.verd
    res = res.drop('verd', axis=1)
    
    d_train = xgb.DMatrix(teste, label=y_train_teste)
    cross_validation = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100)
    
    d_train1 = xgb.DMatrix(res, label=y_train_resto)
    cross_validation1 = xgb.cv(xgb_params, d_train1, num_boost_round=1000, early_stopping_rounds=100) 

     
    num_boost_rounds = len(cross_validation)
    clf = xgb.train(xgb_params, d_train, num_boost_round=num_boost_rounds, verbose_eval=10)    
    d_train = xgb.DMatrix(teste)
    pred = clf.predict(d_train)

    num_boost_rounds1 = len(cross_validation1)
    clf1 = xgb.train(xgb_params, d_train1, num_boost_round=num_boost_rounds1, verbose_eval=10)    
    d_train1 = xgb.DMatrix(res)
    pred1 = clf1.predict(d_train1)

    preditos.append(pd.DataFrame(pred))
    verdadeiros.append(y_train_teste)

    preditos_1.append(pd.DataFrame(pred1))
    verdadeiros_1.append(y_train_resto)


    preditos = pd.concat(preditos, axis=0)
    verdadeiros = pd.concat(verdadeiros, axis=0)

    preditos_1 = pd.concat(preditos_1, axis=0)
    verdadeiros_1 = pd.concat(verdadeiros_1, axis=0)


    preditos_final = pd.concat([preditos,preditos_1], axis=0)
    verdadeiros_final = pd.concat([verdadeiros,verdadeiros_1], axis=0)
    #from sklearn.metrics import mean_absolute_error
    #print "Valor obtido_final_prevendo com todos os dados:{}".format(mean_absolute_error(verdadeiros_final,preditos_final))


#Validation
################################################

    if 'NaN' not in tudo:
        ind = validacao[validacao.regionidcity.isin(tudo)].index
        teste = validacao.loc[ind]
        res = validacao.loc[list(set(validacao.index) - set(ind))]
    if 'NaN' in tudo :
        index1 = validacao.regionidcity.index[validacao.regionidcity.apply(np.isnan)]
        ind = validacao[validacao.regionidcity.isin(tudo)].index
        x_train_interm = validacao.loc[index1]
        teste = validacao.loc[ind]
        teste = pd.concat([x_train_interm, teste] , axis=0)
        tudo = np.concatenate([index1,ind])
        
        res = validacao.loc[list(set(validacao.index) - set(tudo))]

    y_train_teste = teste.verd
    teste = teste.drop('verd', axis=1)
    
    y_train_resto = res.verd
    res = res.drop('verd', axis=1)

    d_train = xgb.DMatrix(teste)
    d_train1 = xgb.DMatrix(res)

    pred_1 = clf.predict(d_train)
    pred_2 = clf1.predict(d_train1)

    #print "Valor obtido_final_teste:{}".format(mean_absolute_error(y_train_teste,pred_1))
    #print "Valor obtido_final_res:{}".format(mean_absolute_error(y_train_resto,pred_2))
    
    preditos1 = []
    verdadeiros1 = []

    preditos_2 = []
    verdadeiros_2 = []

    preditos1.append(pd.DataFrame(pred_1))
    verdadeiros1.append(y_train_teste)

    preditos_2.append(pd.DataFrame(pred_2))
    verdadeiros_2.append(y_train_resto)


    preditos1 = pd.concat(preditos1, axis=0)
    verdadeiros1 = pd.concat(verdadeiros1, axis=0)

    preditos_2 = pd.concat(preditos_2, axis=0)
    verdadeiros_2 = pd.concat(verdadeiros_2, axis=0)


    preditos_final = pd.concat([preditos1,preditos_2], axis=0)
    verdadeiros_final = pd.concat([verdadeiros1,verdadeiros_2], axis=0)
#from sklearn.metrics import mean_absolute_error
    #print "Valor obtido_final_prevendo com todos os dados em validacao:{}".format(mean_absolute_error(verdadeiros_final,preditos_final))   

    
    
    
    res_final_7_temp_1rodada[str(vazio)] = mean_absolute_error(verdadeiros_final,preditos_final)
    
    
    
    
    '''if mean_absolute_error(verdadeiros_final,preditos_final) >= pontuacao:
       vazio.remove(i)
    else:
       pontuacao = mean_absolute_error(verdadeiros_final,preditos_final)
        
    res_final_7_final_1rodada[str(vazio)] = mean_absolute_error(verdadeiros_final,preditos_final)
    print "CV-final obtido:{}".format(mean_absolute_error(verdadeiros_final,preditos_final)) '''   

    tudo_final.append(res_final_7_final_1rodada)
  #np.random.shuffle(cols)

###########################################################
for j in range(0,15):
  tudo = sorted(tudo_final[j], key=tudo_final[j].get)[0]
  tudo = tudo.replace("[" , "")
  tudo = tudo.replace("]" , "")
  tudo = tudo.split(', ')
  for i in range(0,len(tudo)):
      if tudo[i] == "'NaN'":
         tudo[i] = 'NaN'
      else:
          
         tudo[i] = float(tudo[i])

  if 'NaN' not in tudo:
        ind = x_train1[x_train1.regionidcity.isin(tudo)].index
        teste = x_train1.iloc[ind]
        res = x_train1.iloc[list(set(x_train1.index) - set(ind))]
  if 'NaN' in tudo :
        index1 = x_train1.regionidcity.index[x_train1.regionidcity.apply(np.isnan)]
        ind = x_train1[x_train1.regionidcity.isin(tudo)].index
        x_train_interm = x_train1.iloc[index1]
        teste = x_train1.iloc[ind]
        teste = pd.concat([x_train_interm, teste] , axis=0)
        tudo = np.concatenate([index1,ind])
        
        res = x_train1.iloc[list(set(x_train1.index) - set(tudo))]
        
        
  

  y_train_teste = teste.verd
  teste = teste.drop('verd', axis=1)
    
  y_train_resto = res.verd
  res = res.drop('verd', axis=1)
    
  d_train = xgb.DMatrix(teste, label=y_train_teste)
  cross_validation = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100)

  d_train = xgb.DMatrix(res, label=y_train_resto)
  cross_validation1 = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100)
  print "Valor obtido:{}".format((np.array(cross_validation[-1:]['test-mae-mean']) + np.array(cross_validation1[-1:]['test-mae-mean'])) / 2)

################################################################



for i in poss:
   if 'NaN' in i:
      index = x_train1.regionidcity.index[x_train1.regionidcity.apply(np.isnan)]
      x_train_interm = x_train1.iloc[index]
      teste = x_train1[x_train1.regionidcity.isin(i)]
      teste = pd.concat([x_train_interm, teste] , axis=0)   
   
   if 'NaN' not in i:
      teste = x_train1[x_train1.regionidcity.isin(i)] 
       
    
   y_train_10 = teste.verd
   teste = teste.drop('verd', axis=1)
      
   '''plt.plot(teste.pred,teste.verd, 'o')
   plt.xlabel("pred")
   plt.ylabel("verdadeiro")
   plt.title(str(i))
   plt.show()
   plt.plot(teste.diff1, 'o')
   plt.title("residuo" + str(i))
   plt.show()  '''
   
   #train_columns = x_train.columns
   d_train = xgb.DMatrix(teste, label=y_train_10)

   cross_validation = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100)

   res_final_7[str(i)] = np.array(cross_validation[-1:]['test-mae-mean'])
   #num_boost_rounds = len(cross_validation)

   print "CV obtido:{}".format(np.array(cross_validation[-1:]['test-mae-mean']))
   #clf = xgb.train(xgb_params, d_train, num_boost_round=num_boost_rounds, verbose_eval=10)    
   #modelos[str(i)] = clf
   #d_train = xgb.DMatrix(teste)
   #preditos.append(pd.DataFrame(clf.predict(d_train)))
   #verdadeiros.append(y_train_10)




'''res_ord = res_final_7.copy()
res_ord_names = sorted(res_ord, key=res_ord.get)
list_final = []
list_final_1 = {}
b = 0
for i in res_ord_names:
    print "Analisando:{} de {}".format(b,len(res_ord_names))
    
    tg = res_ord_names[b]
    tg = tg.replace("(", "")
    tg = tg.replace(")", "")
    tg = tg.split(', ')
    if tg[0] != "'NaN'":
       tg[0] = float(tg[0])
    else:
       tg[0] = 'NaN'
       
    if tg[1] != "'NaN'":
       tg[1] = float(tg[1])
    else:
       tg[1] = 'NaN'
       
    if tg[2] != "'NaN'":
       tg[2] = float(tg[2])
    else:
       tg[2] = 'NaN'
         
         
    #tg = set(tg)
    if len(set(tg).intersection(set(list_final))) == 0:
       list_final.append(tg[0])
       list_final.append(tg[1])
       list_final.append(tg[2])
       list_final_1[i] = res_ord[i]
       
    b += 1   '''
    

##########################################################3  
test_columns = list(train_columns)
test_columns.append('ParcelId')
 
listando1 = []
listando1.append([2])
listando1.append([3])
listando1.append([2.5])
listando1.append([1.5])
listando1.append([1, 3.5, 4, 5, 0, 4.5, 6, 7, 8, 5.5, 6.5, 9, 10, 7.5, 11, 12, 8.5, 20,
                  13, 9.5, 14, 16, 15,0.5, 18,10.50,17,1.75, 11.50, 19,12.5, 14.5, 19.50])   

print("working on test...")

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
df_test['area_finishedfloor1squarefeet_bedroomcnt'] = df_test['finishedfloor1squarefeet'] / df_test['bedroomcnt']
df_test['finishedsquarefeet15_regionidcity'] = df_test['regionidcity'] / df_test['finishedsquarefeet15']


x_test = df_test[test_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)


base_total = []
######################################################################
x_test_interm = x_test[x_test.bathroomcnt.isin(listando1[0])] 
d_test = xgb.DMatrix(x_test_interm[train_columns])
print('Predicting on test ...')
p_test = modelos['[2]'].predict(d_test)   
base = pd.concat([pd.DataFrame(x_test_interm.ParcelId.reset_index(drop=True)), pd.DataFrame(p_test)], axis=1)   
base_total.append(base)

#################################################################################
x_test_interm = x_test[x_test.bathroomcnt.isin(listando1[1])] 
d_test = xgb.DMatrix(x_test_interm[train_columns])
print('Predicting on test ...')
p_test = modelos['[3]'].predict(d_test)   
base = pd.concat([pd.DataFrame(x_test_interm.ParcelId.reset_index(drop=True)), pd.DataFrame(p_test)], axis=1)   
base_total.append(base)
##################################################.
index = x_test.bathroomcnt.index[x_test.bathroomcnt.apply(np.isnan)]
x_test_interm1 = x_test.iloc[index]
x_test_interm = x_test[x_test.bathroomcnt.isin(listando1[2])] 
x_test_interm = pd.concat([x_test_interm, x_test_interm1] , axis=0)
d_test = xgb.DMatrix(x_test_interm[train_columns])
print('Predicting on test ...')
p_test = modelos['[2.5]'].predict(d_test)   
base = pd.concat([pd.DataFrame(x_test_interm.ParcelId.reset_index(drop=True)), pd.DataFrame(p_test)], axis=1)   
base_total.append(base)
################################################################3

x_test_interm = x_test[x_test.bathroomcnt.isin(listando1[3])] 
d_test = xgb.DMatrix(x_test_interm[train_columns])
print('Predicting on test ...')
p_test = modelos['[1.5]'].predict(d_test)   
base = pd.concat([pd.DataFrame(x_test_interm.ParcelId.reset_index(drop=True)), pd.DataFrame(p_test)], axis=1)   
base_total.append(base)

####################################################################

x_test_interm = x_test[x_test.bathroomcnt.isin(listando1[4])]
d_test = xgb.DMatrix(x_test_interm[train_columns])
print('Predicting on test ...')
p_test = modelos['[1, 3.5, 4, 5, 0, 4.5, 6, 7, 8, 5.5, 6.5, 9, 10, 7.5, 11, 12, 8.5, 20]'].predict(d_test)   
base = pd.concat([pd.DataFrame(x_test_interm.ParcelId.reset_index(drop=True)), pd.DataFrame(p_test)], axis=1)   
base_total.append(base)

base_final = pd.concat(base_total)
base_final.columns = ['ParcelId', 'score']
base_final = base_final.reset_index(drop=True)

sub = pd.read_csv('sample_submission.csv')

sub['ParcelId'] = base_final.ParcelId
   
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = base_final.score
       
print('Writing csv ...')
sub.to_csv('xgb_starter.csv', index=False, float_format='%.4f') 

#####################################################################

final = []
contando = 0
for i in cols:
  print "contando {}".format(contando)
  contando += 1
  interm = {}  
  for j in poss[0:52635]:
    ind, = np.where(np.array(j) == i)
    if ind == 0:
       interm[j] = res_ord[str(j)]
  final.append(interm)   


gt = {}
c = 0
for i in final[0:170]:
    minimo = min(i, key=i.get)
    print "Minimo e:{}".format(minimo)
    print "Valor:{}".format(res_ord[str(minimo)])
    print "Maximo:{}".format(c)
    c += 1
    
    gt[minimo] = res_ord[str(minimo)]

contand = 0
repetidos = {}    
for i in gt:
  interm = {}
  for j in gt:
      if len(set(i).intersection(set(j))) != 0:
          #print("pronto")
          #contand += 1
          interm[j] = res_ord[str(j)]
  repetidos[i] = interm        
      
#######################################################################

'''for i in range(0,15):
    funcoes.drodar_valores(x_train1.copy(),validacao.copy(),tudo_final,i,xgb_params)
    print("################################################")'''
    
name = 'regionidcity'   
preditos = []
verdadeiros = []

preditos_1 = []
verdadeiros_1 = []
     
tudo1 = sorted(tudo_final[3], key=tudo_final[3].get)[0]

tudo1 = melhor

'''tudo1 = tudo1.replace("[" , "")
tudo1 = tudo1.replace("]" , "")
tudo1 = tudo1.split(', ')




for i in range(0,len(tudo1)):
      if tudo1[i] == 'NaN':
         tudo1[i] = 'NaN'
      else:
          
         tudo1[i] = float(tudo1[i])'''

tudo = list(tudo1)
#tudo.append(46298) # 
#tudo.append(16764) 
#tudo.append(19177.0)
#tudo.append(47568)
#tudo.append(396550)
    #tudo.append(1124)
#tudo.append(24812)
#tudo.append(46178)
#tudo.append(396053)
#tudo.append(118875)
#tudo.append(51861)
#tudo.append(39308)
#tudo.append(25468)
#tudo.append(3491)
#tudo.append(13693)
#tudo.append(21412)
#tudo.append(25974)
#tudo.append(30399)
#tudo.append(118694)
#tudo.append(40227)
#tudo.append(42091)
#tudo.append(14906)
#np.random.choice(cols,j, replace=False)

if 'NaN' not in tudo:
        ind = treino[treino[name].isin(tudo)].index
        selection_index = treino.loc[ind]
        selection_index_rest = treino.loc[list(set(treino.index) - set(ind))]
if 'NaN' in tudo:
        index1 = treino[name].index[treino[name].apply(np.isnan)]
        ind = treino[treino[name].isin(tudo)].index
        x_train_interm = treino.loc[index1]
        selection_index = treino.loc[ind]
        selection_index = pd.concat([x_train_interm, selection_index] , axis=0)
        tudo = np.concatenate([index1,ind])
        
        selection_index_rest = treino.loc[list(set(treino.index) - set(tudo))]
        
        
  

y_train_teste = selection_index.logerror
selection_index = selection_index.drop('logerror', axis=1)
    
y_train_resto = selection_index_rest.logerror
selection_index_rest = selection_index_rest.drop('logerror', axis=1)
    
d_train = xgb.DMatrix(selection_index, label=y_train_teste)
cross_validation = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100)

d_train1 = xgb.DMatrix(selection_index_rest, label=y_train_resto)
cross_validation1 = xgb.cv(xgb_params, d_train1, num_boost_round=1000, early_stopping_rounds=100) 
print "Valor obtido_cross_validation_teste:{}".format(np.array(cross_validation[-1:]['test-mae-mean']))
print "Valor obtido_cross_validation_res:{}".format(np.array(cross_validation1[-1:]['test-mae-mean']))

print "Valor obtido_cross_validation_somado_dividido_por_2:{}".format((np.array(cross_validation[-1:]['test-mae-mean']) + np.array(cross_validation1[-1:]['test-mae-mean'])) / 2)


num_boost_rounds = len(cross_validation)
clf = xgb.train(xgb_params, d_train, num_boost_round=num_boost_rounds, verbose_eval=10)    
d_train = xgb.DMatrix(selection_index)
pred = clf.predict(d_train)

num_boost_rounds1 = len(cross_validation1)
clf1 = xgb.train(xgb_params, d_train1, num_boost_round=num_boost_rounds1, verbose_eval=10)    
d_train1 = xgb.DMatrix(selection_index_rest)
pred1 = clf1.predict(d_train1)

preditos.append(pd.DataFrame(pred))
verdadeiros.append(y_train_teste)

preditos_1.append(pd.DataFrame(pred1))
verdadeiros_1.append(y_train_resto)


preditos = pd.concat(preditos, axis=0)
verdadeiros = pd.concat(verdadeiros, axis=0)

preditos_1 = pd.concat(preditos_1, axis=0)
verdadeiros_1 = pd.concat(verdadeiros_1, axis=0)


preditos_final = pd.concat([preditos,preditos_1], axis=0)
verdadeiros_final = pd.concat([verdadeiros,verdadeiros_1], axis=0)
#from sklearn.metrics import mean_absolute_error
print "Valor obtido_final_prevendo com todos os dados:{}".format(mean_absolute_error(verdadeiros_final,preditos_final))


#Validation
################################################

if 'NaN' not in tudo:
        ind = validacao[validacao[name].isin(tudo)].index
        selection_index = validacao.loc[ind]
        selection_index_rest = validacao.loc[list(set(validacao.index) - set(ind))]
if 'NaN' in tudo:
        index1 = validacao[name].index[validacao[name].apply(np.isnan)]
        ind = validacao[validacao[name].isin(tudo)].index
        x_train_interm = validacao.loc[index1]
        selection_index = validacao.loc[ind]
        selection_index = pd.concat([x_train_interm, selection_index] , axis=0)
        tudo = np.concatenate([index1,ind])
        
        selection_index_rest = validacao.loc[list(set(validacao.index) - set(tudo))]

y_train_teste = selection_index.logerror
selection_index = selection_index.drop('logerror', axis=1)
    
y_train_resto = selection_index_rest.logerror
selection_index_rest = selection_index_rest.drop('logerror', axis=1)

d_train = xgb.DMatrix(selection_index)
d_train1 = xgb.DMatrix(selection_index_rest)

pred_1 = clf.predict(d_train)
pred_2 = clf1.predict(d_train1)


print "Valor obtido_final_teste:{}".format(mean_absolute_error(y_train_teste,pred_1))
print "Valor obtido_final_res:{}".format(mean_absolute_error(y_train_resto,pred_2))
    
preditos1 = []
verdadeiros1 = []

preditos_2 = []
verdadeiros_2 = []

preditos1.append(pd.DataFrame(pred_1))
verdadeiros1.append(y_train_teste)

preditos_2.append(pd.DataFrame(pred_2))
verdadeiros_2.append(y_train_resto)


preditos1 = pd.concat(preditos1, axis=0)
verdadeiros1 = pd.concat(verdadeiros1, axis=0)

preditos_2 = pd.concat(preditos_2, axis=0)
verdadeiros_2 = pd.concat(verdadeiros_2, axis=0)


preditos_final = pd.concat([preditos1,preditos_2], axis=0)
verdadeiros_final = pd.concat([verdadeiros1,verdadeiros_2], axis=0)
#from sklearn.metrics import mean_absolute_error
print "Valor obtido_final_prevendo com todos os dados em validacao:{}".format(mean_absolute_error(verdadeiros_final,preditos_final))


#Treinar modelo final com todos os dados##############################3
#x_train = x_train.drop(['transaction_month','transaction_week'], axis=1)

if 'NaN' not in tudo:
        ind = x_train[x_train[name].isin(tudo)].index
        selection_index = x_train.loc[ind]
        selection_index_rest = x_train.loc[list(set(x_train.index) - set(ind))]
if 'NaN' in tudo :
        index1 = x_train[name].index[x_train[name].apply(np.isnan)]
        ind = x_train[x_train[name].isin(tudo)].index
        x_train_interm = x_train.loc[index1]
        selection_index = x_train.loc[ind]
        selection_index = pd.concat([x_train, selection_index] , axis=0)
        tudo = np.concatenate([index1,ind])
        
        selection_index_rest = x_train.loc[list(set(x_train.index) - set(tudo))]
        
        
  
y_train_teste = selection_index.logerror
selection_index = selection_index.drop(['logerror','transaction_month', 'transaction_week'], axis=1)
    
y_train_resto = selection_index_rest.logerror
selection_index_rest = selection_index_rest.drop(['logerror','transaction_month', 'transaction_week'], axis=1)
    
d_train = xgb.DMatrix(selection_index, label=y_train_teste)
cross_validation = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100)

d_train1 = xgb.DMatrix(selection_index_rest, label=y_train_resto)
cross_validation1 = xgb.cv(xgb_params, d_train1, num_boost_round=1000, early_stopping_rounds=100)
#print "Valor obtido_cross_validation_teste:{}".format(np.array(cross_validation[-1:]['test-mae-mean']))
#print "Valor obtido_cross_validation_res:{}".format(np.array(cross_validation1[-1:]['test-mae-mean']))

#print "Valor obtido_cross_validation_somado_dividido_por_2:{}".format((np.array(cross_validation[-1:]['test-mae-mean']) + np.array(cross_validation1[-1:]['test-mae-mean'])) / 2)


num_boost_rounds = len(cross_validation)
clf = xgb.train(xgb_params, d_train, num_boost_round=num_boost_rounds, verbose_eval=10)    
#d_train = xgb.DMatrix(teste)
#pred = clf.predict(d_train)

num_boost_rounds1 = len(cross_validation1)
clf1 = xgb.train(xgb_params, d_train1, num_boost_round=num_boost_rounds1, verbose_eval=10)    
#d_train1 = xgb.DMatrix(res)
#pred1 = clf1.predict(d_train1)



    
####################################       

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')


df_test['area_finishedfloor1squarefeet_bedroomcnt'] = df_test['finishedfloor1squarefeet'] / df_test['bedroomcnt']
df_test['finishedsquarefeet15_regionidcity'] = df_test['regionidcity'] / df_test['finishedsquarefeet15']

test_columns = list(train_columns)
test_columns.append('ParcelId')
x_test = df_test[test_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

ind = x_test[x_test[name].isin(tudo)].index
teste = x_test.iloc[ind]
index_resto = list(set(x_test.index) - set(ind))
res = x_test.iloc[index_resto]

d_test = xgb.DMatrix(teste[train_columns])
d_test1 = xgb.DMatrix(res[train_columns])

print('Predicting on test ...')

p_test = clf.predict(d_test)
p_test1 = clf1.predict(d_test1)

final_1 = pd.concat([pd.DataFrame(p_test),teste.ParcelId.reset_index(drop=True)], axis=1)
final_2 = pd.concat([pd.DataFrame(p_test1),res.ParcelId.reset_index(drop=True)], axis=1)

junto = pd.concat([final_1, final_2], axis=0)

junto.columns = ['score', 'ParcelId']


sub = pd.read_csv('sample_submission.csv')
sub.ParcelId = junto.ParcelId.reset_index(drop=True)
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = junto.score.reset_index(drop=True)
       
print('Writing csv ...')
sub.to_csv('xgb_starter.csv', index=False, float_format='%.4f') 

################################################################

name = 'regionidcity'
#print "Valor obtido:{}".format((np.array(cross_validation[-1:]['train-mae-mean']) + np.array(cross_validation1[-1:]['train-mae-mean'])) / 2)
final_opcoes1 = []
final_opcoes = [] 
for j in range(20,100,5):
  #res_final_7_final_1rodada = {}
  #res_final_7_temp_1rodada = {}
  tudo_final = []
  tudo_final1 = []
  #verdadeiros = []
  #preditos = []
  #modelos = {}
  #vazio = []
  pontuacao = 1000000000000000000000000000000000000000000000
  for i in range(0,10):
    res_final_7_final_1rodada = {}
    res_final_7_final_2rodada = {}

    vazio = np.random.choice(cols,j, replace=False)
    vazio = list(vazio)

    for p in range(0,len(vazio)):
      if vazio[p] == 'NaN':
         vazio[p] = 'NaN'
      else:
          
         vazio[p] = float(vazio[p])

    
    
    preditos = []
    verdadeiros = []

    preditos_1 = []
    verdadeiros_1 = []
    #vazio = tudo
    #vazio.append(i)
    if 'NaN' not in vazio:
        ind = treino[treino[name].isin(vazio)].index
        selection_index = treino.loc[ind]
        selection_index_rest = treino.loc[list(set(treino.index) - set(ind))]
    if 'NaN' in vazio:
        index1 = treino[name].index[treino[name].apply(np.isnan)]
        ind = treino[treino[name].isin(vazio)].index
        x_train_interm = treino.loc[index1]
        selection_index = treino.loc[ind]
        selection_index = pd.concat([x_train_interm, selection_index] , axis=0)
        tudo = np.concatenate([index1,ind])
        
        selection_index_rest = treino.loc[list(set(treino.index) - set(tudo))]
        
        
    y_train_teste = selection_index.logerror
    selection_index = selection_index.drop('logerror', axis=1)
    
    y_train_resto = selection_index_rest.logerror
    selection_index_rest = selection_index_rest.drop('logerror', axis=1)
    
    d_train = xgb.DMatrix(selection_index, label=y_train_teste)
    cross_validation = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100)
    
    d_train1 = xgb.DMatrix(selection_index_rest, label=y_train_resto)
    cross_validation1 = xgb.cv(xgb_params, d_train1, num_boost_round=1000, early_stopping_rounds=100) 

     
    num_boost_rounds = len(cross_validation)
    clf = xgb.train(xgb_params, d_train, num_boost_round=num_boost_rounds, verbose_eval=10)    
    d_train = xgb.DMatrix(selection_index)
    pred = clf.predict(d_train)

    num_boost_rounds1 = len(cross_validation1)
    clf1 = xgb.train(xgb_params, d_train1, num_boost_round=num_boost_rounds1, verbose_eval=10)    
    d_train1 = xgb.DMatrix(selection_index_rest)
    pred1 = clf1.predict(d_train1)

    preditos.append(pd.DataFrame(pred))
    verdadeiros.append(y_train_teste)

    preditos_1.append(pd.DataFrame(pred1))
    verdadeiros_1.append(y_train_resto)


    preditos = pd.concat(preditos, axis=0)
    verdadeiros = pd.concat(verdadeiros, axis=0)

    preditos_1 = pd.concat(preditos_1, axis=0)
    verdadeiros_1 = pd.concat(verdadeiros_1, axis=0)


    preditos_final = pd.concat([preditos,preditos_1], axis=0)
    verdadeiros_final = pd.concat([verdadeiros,verdadeiros_1], axis=0)
    #from sklearn.metrics import mean_absolute_error
    print "Valor obtido_final_prevendo com todos os dados:{}".format(mean_absolute_error(verdadeiros_final,preditos_final))
    res_final_7_final_2rodada[str(vazio)] = mean_absolute_error(verdadeiros_final,preditos_final)


#Validation
################################################

    if 'NaN' not in vazio:
        ind = validacao[validacao[name].isin(vazio)].index
        selection_index = validacao.loc[ind]
        selection_index_rest = validacao.loc[list(set(validacao.index) - set(ind))]
    if 'NaN' in vazio :
        index1 = validacao[name].index[validacao[name].apply(np.isnan)]
        ind = validacao[validacao[name].isin(vazio)].index
        x_train_interm = validacao.loc[index1]
        selection_index = validacao.loc[ind]
        selection_index = pd.concat([x_train_interm, selection_index] , axis=0)
        tudo = np.concatenate([index1,ind])
        
        selection_index_rest = validacao.loc[list(set(validacao.index) - set(tudo))]

    y_train_teste = selection_index.logerror
    selection_index = selection_index.drop('logerror', axis=1)
    
    y_train_resto = selection_index_rest.logerror
    selection_index_rest = selection_index_rest.drop('logerror', axis=1)

    d_train = xgb.DMatrix(selection_index)
    d_train1 = xgb.DMatrix(selection_index_rest)

    pred_1 = clf.predict(d_train)
    pred_2 = clf1.predict(d_train1)

    #print "Valor obtido_final_teste:{}".format(mean_absolute_error(y_train_teste,pred_1))
    #print "Valor obtido_final_res:{}".format(mean_absolute_error(y_train_resto,pred_2))
    
    preditos1 = []
    verdadeiros1 = []

    preditos_2 = []
    verdadeiros_2 = []

    preditos1.append(pd.DataFrame(pred_1))
    verdadeiros1.append(y_train_teste)

    preditos_2.append(pd.DataFrame(pred_2))
    verdadeiros_2.append(y_train_resto)


    preditos1 = pd.concat(preditos1, axis=0)
    verdadeiros1 = pd.concat(verdadeiros1, axis=0)

    preditos_2 = pd.concat(preditos_2, axis=0)
    verdadeiros_2 = pd.concat(verdadeiros_2, axis=0)


    preditos_final = pd.concat([preditos1,preditos_2], axis=0)
    verdadeiros_final = pd.concat([verdadeiros1,verdadeiros_2], axis=0)
    #from sklearn.metrics import mean_absolute_error
    #print "Valor obtido_final_prevendo com todos os dados em validacao:{}".format(mean_absolute_error(verdadeiros_final,preditos_final))   
    res_final_7_final_1rodada[str(vazio)] = mean_absolute_error(verdadeiros_final,preditos_final)

    
    
    
    #res_final_7_temp_1rodada[str(vazio)] = mean_absolute_error(verdadeiros_final,preditos_final)
    
    
    
    
    '''if mean_absolute_error(verdadeiros_final,preditos_final) >= pontuacao:
       vazio.remove(i)
    else:
       pontuacao = mean_absolute_error(verdadeiros_final,preditos_final)
        
    res_final_7_final_1rodada[str(vazio)] = mean_absolute_error(verdadeiros_final,preditos_final)'''
    print "CV-final obtido:{}".format(mean_absolute_error(verdadeiros_final,preditos_final))  

    tudo_final.append(res_final_7_final_1rodada)
    tudo_final1.append(res_final_7_final_2rodada)

  final_opcoes.append(tudo_final)
  final_opcoes1.append(tudo_final1)  
  
  #np.random.shuffle(cols)
  
###################################################################

teste1 = {}
for j in range(0,10):
  teste = {}
  for i in range(0,len(final_opcoes[j])):
      teste[final_opcoes[j][i].keys()[0]] = final_opcoes[j][i].values()[0]
      
  tudo = sorted(teste, key=teste.get)[0]
  #print "Melhor- CV-final obtido:{}-variaveis{}".format(teste[tudo],tudo)  
   
  teste1[tudo] = teste[tudo]
  
  
melhor = sorted(teste1, key=teste1.get)[1]
print(teste1[melhor]) 
  
#np.random.choice(cols,j, replace=False)

#############################################################

melhor = [29712.0, 13716.0, 52842.0, 40009.0, 32927.0, 15554.0, 22827.0, 6021.0, 10774.0, 34780.0, 118914.0, 
6285.0, 51617.0, 36502.0, 272578.0, 45602.0, 114834.0, 40081.0, 54212.0, 118878.0, 54053.0, 10723.0,
 118880.0, 55753.0, 33836.0, 53636.0, 5534.0, 12447.0, 46080.0, 18098.0, 24832.0, 113576.0, 14542.0, 
27491.0, 396551.0, 10389.0, 26965.0, 30267.0, 9840.0, 25458.0, 118994.0, 45457.0, 4406.0, 27110.0, 47019.0,
26483.0, 25218.0, 33837.0, 52650.0, 114828.0, 1124 ]

res_final_7_final_100rodada = {}
res_final_7_final_200rodada = {}

for i in resto:
    #res_final_7_final_1rodada = {}
    #res_final_7_final_2rodada = {}

    #vazio = np.random.choice(cols,j, replace=False)
    #vazio = list(vazio)
    preditos = []
    verdadeiros = []

    preditos_1 = []
    verdadeiros_1 = []
     
    #tudo1 = sorted(tudo_final[3], key=tudo_final[3].get)[0]

    vazio = list(melhor)

    '''vazio = vazio.replace("[" , "")
    vazio = vazio.replace("]" , "")
    vazio = vazio.split(', ')


    for p in range(0,len(vazio)):
      if vazio[p] == 'NaN':
         vazio[p] = 'NaN'
      else:
          
         vazio[p] = float(vazio[p])'''

    
    
   
    #vazio = tudo
    vazio.append(i)
    if 'NaN' not in vazio:
        ind = treino[treino[name].isin(vazio)].index
        selection_index = treino.loc[ind]
        selection_index_rest = treino.loc[list(set(treino.index) - set(ind))]
    if 'NaN' in vazio:
        index1 = treino[name].index[treino[name].apply(np.isnan)]
        ind = treino[treino[name].isin(vazio)].index
        x_train_interm = treino.loc[index1]
        selection_index = treino.loc[ind]
        selection_index = pd.concat([x_train_interm, selection_index] , axis=0)
        tudo = np.concatenate([index1,ind])
        
        selection_index_rest = treino.loc[list(set(treino.index) - set(tudo))]
        
        
    y_train_teste = selection_index.logerror
    selection_index = selection_index.drop('logerror', axis=1)
    
    y_train_resto = selection_index_rest.logerror
    selection_index_rest = selection_index_rest.drop('logerror', axis=1)
    
    d_train = xgb.DMatrix(selection_index, label=y_train_teste)
    cross_validation = xgb.cv(xgb_params, d_train, num_boost_round=1000, early_stopping_rounds=100)
    
    d_train1 = xgb.DMatrix(selection_index_rest, label=y_train_resto)
    cross_validation1 = xgb.cv(xgb_params, d_train1, num_boost_round=1000, early_stopping_rounds=100) 

     
    num_boost_rounds = len(cross_validation)
    clf = xgb.train(xgb_params, d_train, num_boost_round=num_boost_rounds, verbose_eval=10)    
    d_train = xgb.DMatrix(selection_index)
    pred = clf.predict(d_train)

    num_boost_rounds1 = len(cross_validation1)
    clf1 = xgb.train(xgb_params, d_train1, num_boost_round=num_boost_rounds1, verbose_eval=10)    
    d_train1 = xgb.DMatrix(selection_index_rest)
    pred1 = clf1.predict(d_train1)

    preditos.append(pd.DataFrame(pred))
    verdadeiros.append(y_train_teste)

    preditos_1.append(pd.DataFrame(pred1))
    verdadeiros_1.append(y_train_resto)


    preditos = pd.concat(preditos, axis=0)
    verdadeiros = pd.concat(verdadeiros, axis=0)

    preditos_1 = pd.concat(preditos_1, axis=0)
    verdadeiros_1 = pd.concat(verdadeiros_1, axis=0)


    preditos_final = pd.concat([preditos,preditos_1], axis=0)
    verdadeiros_final = pd.concat([verdadeiros,verdadeiros_1], axis=0)
    #from sklearn.metrics import mean_absolute_error
    print "Valor obtido_final_prevendo com todos os dados:{}".format(mean_absolute_error(verdadeiros_final,preditos_final))
    res_final_7_final_200rodada[str(vazio)] = mean_absolute_error(verdadeiros_final,preditos_final)


#Validation
################################################

    if 'NaN' not in vazio:
        ind = validacao[validacao[name].isin(vazio)].index
        selection_index = validacao.loc[ind]
        selection_index_rest = validacao.loc[list(set(validacao.index) - set(ind))]
    if 'NaN' in vazio :
        index1 = validacao[name].index[validacao[name].apply(np.isnan)]
        ind = validacao[validacao[name].isin(vazio)].index
        x_train_interm = validacao.loc[index1]
        selection_index = validacao.loc[ind]
        selection_index = pd.concat([x_train_interm, selection_index] , axis=0)
        tudo = np.concatenate([index1,ind])
        
        selection_index_rest = validacao.loc[list(set(validacao.index) - set(tudo))]

    y_train_teste = selection_index.logerror
    selection_index = selection_index.drop('logerror', axis=1)
    
    y_train_resto = selection_index_rest.logerror
    selection_index_rest = selection_index_rest.drop('logerror', axis=1)

    d_train = xgb.DMatrix(selection_index)
    d_train1 = xgb.DMatrix(selection_index_rest)

    pred_1 = clf.predict(d_train)
    pred_2 = clf1.predict(d_train1)

    #print "Valor obtido_final_teste:{}".format(mean_absolute_error(y_train_teste,pred_1))
    #print "Valor obtido_final_res:{}".format(mean_absolute_error(y_train_resto,pred_2))
    
    preditos1 = []
    verdadeiros1 = []

    preditos_2 = []
    verdadeiros_2 = []

    preditos1.append(pd.DataFrame(pred_1))
    verdadeiros1.append(y_train_teste)

    preditos_2.append(pd.DataFrame(pred_2))
    verdadeiros_2.append(y_train_resto)


    preditos1 = pd.concat(preditos1, axis=0)
    verdadeiros1 = pd.concat(verdadeiros1, axis=0)

    preditos_2 = pd.concat(preditos_2, axis=0)
    verdadeiros_2 = pd.concat(verdadeiros_2, axis=0)


    preditos_final = pd.concat([preditos1,preditos_2], axis=0)
    verdadeiros_final = pd.concat([verdadeiros1,verdadeiros_2], axis=0)
    #from sklearn.metrics import mean_absolute_error
    #print "Valor obtido_final_prevendo com todos os dados em validacao:{}".format(mean_absolute_error(verdadeiros_final,preditos_final))   
    res_final_7_final_100rodada[i] = mean_absolute_error(verdadeiros_final,preditos_final)

    
    
    
    #res_final_7_temp_1rodada[str(vazio)] = mean_absolute_error(verdadeiros_final,preditos_final)
    
    
    
    
    '''if mean_absolute_error(verdadeiros_final,preditos_final) >= pontuacao:
       vazio.remove(i)
    else:
       pontuacao = mean_absolute_error(verdadeiros_final,preditos_final)
        
    res_final_7_final_1rodada[str(vazio)] = mean_absolute_error(verdadeiros_final,preditos_final)'''
    print "CV-final obtido:{}".format(mean_absolute_error(verdadeiros_final,preditos_final))  

    #tudo_final.append(res_final_7_final_1rodada)
    #tudo_final1.append(res_final_7_final_2rodada)

  #final_opcoes.append(tudo_final)
  #final_opcoes1.append(tudo_final1) 

###########################################################################

tudo = [29712.0, 13716.0, 52842.0, 40009.0, 32927.0, 15554.0, 22827.0, 6021.0, 10774.0, 34780.0, 118914.0, 
6285.0, 51617.0, 36502.0, 272578.0, 45602.0, 114834.0, 40081.0, 54212.0, 118878.0, 54053.0, 10723.0,
 118880.0, 55753.0, 33836.0, 53636.0, 5534.0, 12447.0, 46080.0, 18098.0, 24832.0, 113576.0, 14542.0, 
27491.0, 396551.0, 10389.0, 26965.0, 30267.0, 9840.0, 25458.0, 118994.0, 45457.0, 4406.0, 27110.0, 47019.0,
26483.0, 25218.0, 33837.0, 52650.0, 114828.0, 1124]


index_treino, index_validacao = funcoes.cria_treino_validacao(x_train)

treino = x_train.iloc[index_treino]
validacao = x_train.iloc[index_validacao]

#y_treino = treino.logerror
treino = treino.drop(['transaction_month','transaction_week'], axis=1)
#y_validacao = validacao.logerror
validacao = validacao.drop(['transaction_month','transaction_week'], axis=1)


name = 'regionidcity'

if 'NaN' not in tudo:
        ind = treino[treino[name].isin(tudo)].index
        selection_index = treino.loc[ind]
        selection_index_rest = treino.loc[list(set(treino.index) - set(ind))]
if 'NaN' in tudo :
        index1 = treino[name].index[treino[name].apply(np.isnan)]
        ind = treino[treino[name].isin(tudo)].index
        x_train_interm = treino.loc[index1]
        selection_index = treino.loc[ind]
        selection_index = pd.concat([treino, selection_index] , axis=0)
        tudo = np.concatenate([index1,ind])
        
        selection_index_rest = treino.loc[list(set(treino.index) - set(tudo))]

y_train = selection_index.logerror.values
selection_index.drop('logerror', axis=1, inplace=True)
#x_train1 = selection_index.values

#####################################

y_train1 = selection_index_rest.logerror.values
selection_index_rest.drop('logerror', axis=1, inplace=True)
#x_train11 = selection_index_rest.values

#####################################


print('Iniciando o treinamento do ensemble') 
#x_train1 = x_train.copy()
         
ntrain = selection_index.shape[0]
SEED = 0 
NFOLDS = 5 
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)


xgb_params_1 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'random_state': 20
                 }  

xgb_params_2 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'random_state' : 30
                 }  
    
xgb_params_3 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'random_state' : 40
                 }  

xgb_params_4 = {
      'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'random_state' : 50
                 }  


xgb_params_5 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'random_state' : 60
                 }  

xgb_params_6 = {
     'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'random_state' : 70
                 }  


numero = np.arange(10,x_train1.shape[1])


xgbo_1 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_1,flag=1)
xgbo_2 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_2,flag=1)
xgbo_3 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_3,flag=1)
xgbo_4 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_4,flag=1)
xgbo_5 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_5,flag=1)
xgbo_6 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_6,flag=1)



print('Iniciando o treinamento dos classificadores de primeiro nivel')
print('Iniciando o Extreme Gradient Boost')
#numero_col_1 = np.random.choice(np.arange(0,44),1)
#col_1 = np.random.choice(numero,numero_col_1)

xgbo_1_oof_train = funcoes.get_oof(xgbo_1, selection_index.values, y_train,ntrain,kf) 

print('Iniciando o Extreme Gradient Boost')
#numero_col_2 = np.random.choice(np.arange(0,44),1)
#col_2 = np.random.choice(numero,numero_col_2)
xgbo_2_oof_train= funcoes.get_oof(xgbo_2,selection_index.values, y_train,ntrain,kf) 
  
                        
print('Iniciando o Extreme Gradient Boost')
#numero_col_3 = np.random.choice(np.arange(0,44),1)
#col_3 = np.random.choice(numero,numero_col_3)
xgbo_3_oof_train_1= funcoes.get_oof(xgbo_3,selection_index.values, y_train,ntrain,kf)
 
  
print('Iniciando o Extreme Gradient Boost')
#numero_col_4 = np.random.choice(np.arange(0,44),1)
#col_4 = np.random.choice(numero,numero_col_4)
xgbo_4_oof_train = funcoes.get_oof(xgbo_4, selection_index.values, y_train,ntrain,kf) #

    
print('Iniciando o Extreme Gradient Boost')
#numero_col_5 = np.random.choice(np.arange(0,44),1)
#col_5 = np.random.choice(numero,numero_col_5)
xgbo_5_oof_train = funcoes.get_oof(xgbo_5,selection_index.values, y_train,ntrain,kf) 
    
print('Iniciando o Extreme Gradient Boost')
#numero_col_6 = np.random.choice(np.arange(0,44),1)
#col_6 = np.random.choice(numero,numero_col_6)
xgbo_6_oof_train = funcoes.get_oof(xgbo_6,x_train1, y_train,ntrain,kf) 

x_train_inter = np.concatenate((xgbo_1_oof_train, xgbo_2_oof_train, xgbo_3_oof_train_1,xgbo_4_oof_train,xgbo_5_oof_train,xgbo_6_oof_train), axis=1)

xgb_params_final = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1
                 }  
    
   
print('segundo nivel')
modelo_final = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_final,flag=1)
xgbo_6_oof_train_final = funcoes.get_oof(modelo_final,x_train_inter, y_train,ntrain,kf)

############################################################
print('Iniciando o treinamento do ensemble') 
#x_train1 = x_train.copy()
         
ntrain = selection_index_rest.shape[0]
SEED = 0 
NFOLDS = 5 
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)


xgb_params_1 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'random_state': 20
                 }  

xgb_params_2 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'random_state' : 30
                 }  
    
xgb_params_3 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'random_state' : 40
                 }  

xgb_params_4 = {
      'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'random_state' : 50
                 }  


xgb_params_5 = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'random_state' : 60
                 }  

xgb_params_6 = {
     'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'random_state' : 70
                 } 

xgbo_11 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_1,flag=1)
xgbo_22 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_2,flag=1)
xgbo_33 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_3,flag=1)
xgbo_44 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_4,flag=1)
xgbo_55 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_5,flag=1)
xgbo_66 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_6, flag=1)


print('Iniciando o treinamento dos classificadores de primeiro nivel')
print('Iniciando o Extreme Gradient Boost')
xgbo_11_oof_train = funcoes.get_oof(xgbo_11, selection_index_rest.values, y_train1,ntrain,kf) # Extra Trees
   
print('Iniciando o Extreme Gradient Boost')
xgbo_22_oof_train= funcoes.get_oof(xgbo_22,selection_index_rest.values, y_train1,ntrain,kf) # Random Forest
                        
print('Iniciando o Extreme Gradient Boost')
xgbo_33_oof_train_1= funcoes.get_oof(xgbo_33,selection_index_rest.values, y_train1,ntrain,kf)
   
print('Iniciando o Extreme Gradient Boost')
xgbo_44_oof_train = funcoes.get_oof(xgbo_44, selection_index_rest.values, y_train1,ntrain,kf) # AdaBoost 
    
print('Iniciando o Extreme Gradient Boost')
xgbo_55_oof_train = funcoes.get_oof(xgbo_55,selection_index_rest.values, y_train1,ntrain,kf) # Gradient Boost
    
print('Iniciando o Extreme Gradient Boost')
xgbo_66_oof_train = funcoes.get_oof(xgbo_66,selection_index_rest.values, y_train1,ntrain,kf) # Extreme Gradient Boost


x_train_inter1 = np.concatenate((xgbo_11_oof_train, xgbo_22_oof_train, xgbo_33_oof_train_1,xgbo_44_oof_train,xgbo_55_oof_train,xgbo_66_oof_train), axis=1)

#X_treino, X_teste, label_treina, label_tes = train_test_split(x_train,y_train,test_size=0.30, random_state=42)
   
xgb_params_final = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'min_child_weight': 7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1
                 } 
print('segundo nivel')
modelo_final1 = ensemble.SklearnHelper(clf=xgb, seed=SEED, params=xgb_params_final,flag=1)
xgbo_66_oof_train_final = funcoes.get_oof(modelo_final1,x_train_inter1, y_train1,ntrain,kf)


#########################modelo final###################################
print('Treinando com todos os dados - parte 1')
funcoes.get_oof_all_data(xgbo_1, selection_index.values, y_train)
funcoes.get_oof_all_data(xgbo_2, selection_index.values, y_train)
funcoes.get_oof_all_data(xgbo_3, selection_index.values, y_train)
funcoes.get_oof_all_data(xgbo_4, selection_index.values, y_train)
funcoes.get_oof_all_data(xgbo_5, selection_index.values, y_train)
funcoes.get_oof_all_data(xgbo_6, selection_index.values, y_train)
funcoes.get_oof_all_data(modelo_final,x_train_inter, y_train)

####################################################################
print('Treinando com todos os dados - parte 2')

funcoes.get_oof_all_data(xgbo_11, selection_index_rest.values, y_train1) 
funcoes.get_oof_all_data(xgbo_22, selection_index_rest.values, y_train1)   
funcoes.get_oof_all_data(xgbo_33, selection_index_rest.values, y_train1)
funcoes.get_oof_all_data(xgbo_44, selection_index_rest.values, y_train1)
funcoes.get_oof_all_data(xgbo_55, selection_index_rest.values, y_train1)
funcoes.get_oof_all_data(xgbo_66, selection_index_rest.values, y_train1)
funcoes.get_oof_all_data(modelo_final1,x_train_inter1, y_train1)

print('Obtendo os CVS da validacao')
funcoes.valida(validacao,tudo,name,xgbo_1,xgbo_2,xgbo_3,xgbo_4,xgbo_5,xgbo_6,modelo_final,xgbo_11,
           xgbo_22,xgbo_33,xgbo_44,xgbo_55,xgbo_66,modelo_final1)

###############################################################

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')


df_test['area_finishedfloor1squarefeet_bedroomcnt'] = df_test['finishedfloor1squarefeet'] / df_test['bedroomcnt']
df_test['finishedsquarefeet15_regionidcity'] = df_test['regionidcity'] / df_test['finishedsquarefeet15']

test_columns = list(train_columns)
test_columns.append('ParcelId')
x_test = df_test[test_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

ind = x_test[x_test[name].isin(tudo)].index
teste = x_test.iloc[ind]
index_resto = list(set(x_test.index) - set(ind))
res = x_test.iloc[index_resto]



#d_test = xgb.DMatrix(teste[train_columns])
#d_test1 = xgb.DMatrix(res[train_columns])

print('Predicting on test ...')


pred_1 = xgbo_1.predict(teste[train_columns].values)

pred_2 = xgbo_2.predict(teste[train_columns].values)

pred_3 = xgbo_3.predict(teste[train_columns].values)

pred_4 = xgbo_4.predict(teste[train_columns].values)

pred_5 = xgbo_5.predict(teste[train_columns].values)

pred_6 = xgbo_6.predict(teste[train_columns].values)

interm = np.concatenate((pred_1.reshape(-1,1), pred_2.reshape(-1,1), pred_3.reshape(-1,1),pred_4.reshape(-1,1),pred_5.reshape(-1,1),pred_6.reshape(-1,1)), axis=1)


pred_final = modelo_final.predict(interm) 



pred_11 = xgbo_11.predict(res[train_columns].values)
pred_22 = xgbo_22.predict(res[train_columns].values)
pred_33 = xgbo_33.predict(res[train_columns].values)
pred_44 = xgbo_44.predict(res[train_columns].values)
pred_55 = xgbo_55.predict(res[train_columns].values)
pred_66 = xgbo_66.predict(res[train_columns].values)

interm_10 = np.concatenate((pred_11.reshape(-1,1), pred_22.reshape(-1,1), pred_33.reshape(-1,1),pred_44.reshape(-1,1),pred_55.reshape(-1,1),pred_66.reshape(-1,1)), axis=1)

pred_final_10 = modelo_final1.predict(interm_10) 




final_1 = pd.concat([pd.DataFrame(pred_final),teste.ParcelId.reset_index(drop=True)], axis=1)
final_2 = pd.concat([pd.DataFrame(pred_final_10),res.ParcelId.reset_index(drop=True)], axis=1)

junto = pd.concat([final_1, final_2], axis=0)

junto.columns = ['score', 'ParcelId']


sub = pd.read_csv('sample_submission.csv')
sub.ParcelId = junto.ParcelId.reset_index(drop=True)
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = junto.score.reset_index(drop=True)
       
print('Writing csv ...')
sub.to_csv('xgb_starter.csv', index=False, float_format='%.4f') 


###############################################################


print(np.mean(xgbo_1.cvs))
print(np.mean(xgbo_2.cvs))
print(np.mean(xgbo_3.cvs))
print(np.mean(xgbo_4.cvs))
print(np.mean(xgbo_5.cvs))
print(np.mean(xgbo_6.cvs))





