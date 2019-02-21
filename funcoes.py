# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import itertools
import pandas as pd
from sklearn.metrics import mean_absolute_error





def ctun_model_dm(data,y_train):
    print("Iniciando o tunning.....")
    dat = xgb.DMatrix(data,y_train)
    
    param_grid = {
    'max_depth':range(3,10,1),
    'min_child_weight':range(1,9,1)
                 }
    
    cvs = {}
    list1 = param_grid['max_depth']
    list2 = param_grid['min_child_weight']
    c = list(itertools.product(list1, list2))
    for i in range(0,len(c)):
        print("Modelo",i)
        
        
        model = {'eta':0.05, 'subsample': 0.7, 'colsample_bytree': 0.7, 
             'objective': 'reg:linear', 'eval_metric': 'mae' , 'silent':1, 'max_depth':c[i][0],
               'min_child_weight':c[i][1]} 
     
        cv_xgb = xgb.cv(model,dat, num_boost_round = 1000, 
                early_stopping_rounds = 100) 
         
        cvs[c[i]] = np.array(cv_xgb[-1:]['test-mae-mean'])
    return cvs

def ctun_model_gama(data,y_train):
    print("Iniciando o tunning gamma.....")
    dat = xgb.DMatrix(data,y_train)
    
    param_grid = {
    'gamma':[i/10.0 for i in range(0,5)]
    
                 }
    
     
    cvs = {}
    list1 = param_grid['gamma']
    #list2 = param_grid['min_child_weight']
    #c = list(itertools.product(list1, list2))
    for i in list1:
        print("Modelo",i)
        
        
        model = {'eta':0.05, 'subsample': 0.7, 'colsample_bytree': 0.7, 
             'objective': 'reg:linear', 'eval_metric': 'mae' , 'silent':1, 'max_depth':6,
               'min_child_weight':8, 'gamma':i} 
     
        cv_xgb = xgb.cv(model,dat, num_boost_round = 1000, 
                early_stopping_rounds = 100) 
         
        
        cvs[i] = np.array(cv_xgb[-1:]['test-mae-mean'])
    return cvs


def ctun_model_colsample(data,y_train):
    print("Iniciando o colsample.....")
    dat = xgb.DMatrix(data,y_train)
    
    param_grid = {
    'subsample':[i/100.0 for i in range(50,95,5)],
    'colsample_bytree':[i/100.0 for i in range(50,95,5)]
      }
    
     
    cvs = {}
    list1 = param_grid['subsample']
    list2 = param_grid['colsample_bytree']
    c = list(itertools.product(list1, list2))
    print(len(c))
    for i in range(0,len(c)):
        print("Modelo",i)
        print("Subsample",i)
        print(c[i][0])
        print("colsample_bytree",i)
        print(c[i][1])
        
        
        model = {'eta':0.05, 'subsample': c[i][0], 'colsample_bytree': c[i][1], 
             'objective': 'reg:linear', 'eval_metric': 'mae' , 'silent':1, 'max_depth':6,
               'min_child_weight':8, 'gamma':0} 
     
        cv_xgb = xgb.cv(model,dat, num_boost_round = 1000, 
                early_stopping_rounds = 100) 
        #print("Pontua:")
        #print(cv_xgb.iloc[cv_xgb.shape[0]-1]['test-rmse-mean'])
        cvs[c[i]] = np.array(cv_xgb[-1:]['test-mae-mean'])

    return cvs


def ctun_model_reg(data,y_train):
    print("Iniciando o tunning gamma.....")
    dat = xgb.DMatrix(data,y_train)
    
    param_grid = {
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
    
                 }
    
     
    cvs = {}
    list1 = param_grid['reg_alpha']
    #list2 = param_grid['min_child_weight']
    #c = list(itertools.product(list1, list2))
    for i in list1:
        print("Modelo",i)
        
        
        model = {'eta':0.05, 'subsample': 0.7, 'colsample_bytree': 0.7, 
             'objective': 'reg:linear', 'eval_metric': 'mae' , 'silent':1, 'max_depth':6,
               'min_child_weight':8, 'gamma':0,'reg_alpha':i} 
     
        cv_xgb = xgb.cv(model,dat, num_boost_round = 1000, 
                early_stopping_rounds = 100) 
         
        cvs[i] = np.array(cv_xgb[-1:]['test-mae-mean'])
    return cvs
  


def excluir_coluna(s):
        lista = []
           
        for i in range(0,len(s)-1):
               
               if s[i] > 0.6:
                  lista.append(i)
                  
      
        return lista
    
def cria_treino_validacao(dados):
    treino = []
    validacao = []
    indices = pd.unique(dados.transaction_month)
    meses = [10,11,12]
    for i in indices:
       ind, = list(np.where(dados.transaction_month == i))
       if i not in meses:
          sele = list(np.random.choice(ind, int(len(ind) * 0.7), replace=False))
          vali = list(set(ind) - set(sele))
          
       else:
          sele = list(np.random.choice(ind, int(len(ind) * 0.3), replace=False))
          vali = list(set(ind) - set(sele))
          
    

       treino.append(sele)
       validacao.append(vali)
       
    x_treino = np.concatenate(treino)
    x_validacao = np.concatenate(validacao)
    
    return x_treino, x_validacao     
       
       
       
def drodar_valores(x_train1,validacao,tudo_final,numero,xgb_params):
    preditos = []
    verdadeiros = []

    preditos_1 = []
    verdadeiros_1 = []
     
    tudo1 = sorted(tudo_final[numero], key=tudo_final[numero].get)[0]



    tudo1 = tudo1.replace("[" , "")
    tudo1 = tudo1.replace("]" , "")
    tudo1 = tudo1.split(', ')



    for i in range(0,len(tudo1)):
        if tudo1[i] == "'NaN'":
           tudo1[i] = 'NaN'
        else:
          
           tudo1[i] = float(tudo1[i])

    tudo = list(tudo1)
    tudo.append(46298) # 0.0645201
    #tudo.append(16764) #pior 0.0645750
    #tudo.append(25218) # 0.0645343
    #tudo.append(5534) # 0.0645590


    if 'NaN' not in tudo:
        ind = x_train1[x_train1.regionidcity.isin(tudo)].index
        teste = x_train1.loc[ind]
        res = x_train1.loc[list(set(x_train1.index) - set(ind))]
    if 'NaN' in tudo :
        index1 = x_train1.regionidcity.index[x_train1.regionidcity.apply(np.isnan)]
        ind = x_train1[x_train1.regionidcity.isin(tudo)].index
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
    print "Valor obtido_cross_validation_teste:{}".format(np.array(cross_validation[-1:]['test-mae-mean']))
    print "Valor obtido_cross_validation_res:{}".format(np.array(cross_validation1[-1:]['test-mae-mean']))

    print "Valor obtido_cross_validation_somado_dividido_por_2:{}".format((np.array(cross_validation[-1:]['test-mae-mean']) + np.array(cross_validation1[-1:]['test-mae-mean'])) / 2)


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
    print "Valor obtido_final_prevendo com todos os dados:{}".format(mean_absolute_error(verdadeiros_final,preditos_final))


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
        
        res = validacao.iloc[list(set(validacao.index) - set(tudo))]

    y_train_teste = teste.verd
    teste = teste.drop('verd', axis=1)
    
    y_train_resto = res.verd
    res = res.drop('verd', axis=1)

    d_train = xgb.DMatrix(teste)
    d_train1 = xgb.DMatrix(res)

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
    

def valida(validacao,tudo,name,xgbo_1,xgbo_2,xgbo_3,xgbo_4,xgbo_5,xgbo_6,modelo_final,xgbo_11,
           xgbo_22,xgbo_33,xgbo_44,xgbo_55,xgbo_66,modelo_final1):
    
    ind = validacao[validacao[name].isin(tudo)].index
    teste = validacao.loc[ind]
    res = validacao.loc[list(set(validacao.index) - set(ind))]

    label_teste = teste.logerror.values
    teste.drop('logerror', axis=1, inplace=True)
    label_res = res.logerror.values
    res.drop('logerror', axis=1, inplace=True)


    pred_1 = xgbo_1.predict(teste.values)
    pred_2 = xgbo_2.predict(teste.values)

    pred_3 = xgbo_3.predict(teste.values)

    pred_4 = xgbo_4.predict(teste.values)

    pred_5 = xgbo_5.predict(teste.values)

    pred_6 = xgbo_6.predict(teste.values)

    interm = np.concatenate((pred_1.reshape(-1,1), pred_2.reshape(-1,1), pred_3.reshape(-1,1),pred_4.reshape(-1,1),pred_5.reshape(-1,1),pred_6.reshape(-1,1)), axis=1)


    pred_final = modelo_final.predict(interm) 



    pred_11 = xgbo_11.predict(res.values)
    pred_22 = xgbo_22.predict(res.values)
    pred_33 = xgbo_33.predict(res.values)
    pred_44 = xgbo_44.predict(res.values)
    pred_55 = xgbo_55.predict(res.values)
    pred_66 = xgbo_66.predict(res.values)

    interm_10 = np.concatenate((pred_11.reshape(-1,1), pred_22.reshape(-1,1), pred_33.reshape(-1,1),pred_44.reshape(-1,1),pred_55.reshape(-1,1),pred_66.reshape(-1,1)), axis=1)

    pred_final_10 = modelo_final1.predict(interm_10) 


    pred_1 = np.concatenate((pred_final.reshape(-1,1),label_teste.reshape(-1,1)), axis=1)
    pred_2 = np.concatenate([pred_final_10.reshape(-1,1),label_res.reshape(-1,1)], axis=1)


    junto = np.concatenate((pred_1,pred_2), axis=0)
    print('CV-primeira particao:')
    print(mean_absolute_error(label_teste, pred_final))
    print('CV-segunda particao:')
    print(mean_absolute_error(label_res, pred_final_10))
    print('particao-toda:')
    print(mean_absolute_error(junto[:,0], junto[:,1]))
    
    
def get_oof(clf, x_train, y_train,ntrain,kf):
    oof_train = np.zeros((ntrain,))
   
    
    
    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
           x_tr = x_train[train_index]
           y_tr = y_train[train_index]
           x_te = x_train[test_index]
           y_te = y_train[test_index]  
           
           #d_train = xgb.DMatrix(x_tr, y_tr)
           #d_valid = xgb.DMatrix(x_te, y_te)
    
           #watchlist = [(d_train, 'train'), (d_valid, 'valid')]

           #model = clf.train(d_train, 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=100)
                        
           #xgb_pred = model.predict(d_test)
    
    
           clf.train(x_tr, y_tr,x_te,y_te)
           
           pred =  clf.predict(x_te)  
           oof_train[test_index] = pred
           clf.save_cv(y_te,pred)
    return oof_train.reshape(-1,1)
           
           
    '''for i, (train_index, test_index) in enumerate(kf):
        #print("passando...")
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        y_te = y_train[test_index]

        clf.train(x_tr, y_tr)
        
        
        pred =  clf.predict(x_te)  
        oof_train[test_index] = pred
        clf.save_cv(y_te,pred)
    return oof_train.reshape(-1,1)'''

def get_oof_all_data(clf, x_train, y_train):
    
    clf.train(x_train, y_train) 
    

    
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)    



def get_oof_encoding(x_train, y_train,ntrain,kf,nome):
    oof_train = np.zeros((ntrain,))
   
    
    #x_train['target'] = y_train
    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
           x_tr = x_train.iloc[train_index]
           y_tr = y_train[train_index]
           x_te = x_train.iloc[test_index]
           #y_te = y_train[test_index]  
           
           x_tr['target'] = y_tr
           train_col_1, teste_col_1 = target_encode(x_tr[nome], 
                         x_te[nome], 
                         target=x_tr.target, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)
           
    
    
           
           
           oof_train[test_index] = teste_col_1
           
    return oof_train.reshape(-1,1)


