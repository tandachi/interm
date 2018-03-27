# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:06:26 2018

@author: rebli
"""

import pandas as pd
import numpy as np
from pandas.core.tools.datetimes import to_datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

def rmsle(y_true,y_pred):
   assert len(y_true) == len(y_pred)
   return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5


def tree(ml_dt):
    dot_data = StringIO()
    export_graphviz(ml_dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())
    graph.write_pdf("tree.pdf")



def ordinal_to_categorical(data, column, categories):
    """Convert an ordinal column to categorical."""
    new_data = data.copy()
    col = new_data[column].astype('category')
    new_data[column] = col.cat.rename_categories(categories)
    return new_data

def date_time(data):
    
    dt = data.datetime.dt
    data['dayofweek'] = dt.dayofweek
    data['day'] = dt.day
    #data['time'] = dt.time
    data['hour'] = dt.hour
    data['year'] = dt.year   
    data['month'] = dt.month
    
    return data


def sep_season(data):
    data_season = dict()

    for i in np.unique(data.season):
       data_season[i] = data[data.season == i]
    return data_season


def get_tr_te(data_tr):
       
    data_te = []
    for month in np.unique(data_tr.month):
       
          data_te.append(data_tr[(data_tr.month == month) & (data_tr.day == data_tr[data_tr.month == month].day.max() )])
          data_tr.drop(data_tr[(data_tr.month == month) & (data_tr.day == data_tr[data_tr.month == month].day.max() )].index, inplace=True)
        
    data_te = pd.concat(data_te, axis=0)
    return data_tr, data_te


def feat_eng(data_season_tr, data_season_te, lista):
    
    for i in lista:
        
       data_season_tr[i]['humidity-diff'] = data_season_tr[i]['humidity'].diff() 
       data_season_te[i]['humidity-diff'] = data_season_te[i]['humidity'].diff()
       
       #data_season_tr[i]['windspeed-diff'] = data_season_tr[i]['windspeed'].diff() 
       #data_season_te[i]['windspeed-diff'] = data_season_te[i]['windspeed'].diff()
       
       data_season_tr[i]['temp'] = data_season_tr[i]['temp'].astype(int) 
       data_season_te[i]['temp'] = data_season_te[i]['temp'].astype(int)
       
       #data_season_tr[i]['atemp'] = data_season_tr[i]['atemp'].astype(int) 
       #data_season_te[i]['atemp'] = data_season_te[i]['atemp'].astype(int)
       
       data_season_tr[i]['diff-temp'] = data_season_tr[i]['atemp'] - data_season_tr[i]['temp']
       data_season_te[i]['diff-temp'] = data_season_te[i]['atemp'] - data_season_te[i]['temp']
       
       data_season_tr[i]['humidty-temp'] = (data_season_tr[i]['humidity'] / data_season_tr[i]['temp']).replace(np.inf, 0)
       data_season_te[i]['humidty-temp'] = (data_season_te[i]['humidity'] / data_season_te[i]['temp']).replace(np.inf, 0)
       
       data_season_tr[i]['shift-1'] = data_season_tr[i]['temp'].shift(1)
       data_season_te[i]['shift-1'] = data_season_te[i]['temp'].shift(1)
       
       data_season_tr[i]['shift-1'] = data_season_tr[i]['shift-1'] - data_season_tr[i]['temp']
       data_season_te[i]['shift-1'] = data_season_te[i]['shift-1'] - data_season_te[i]['temp']
       
       
       #fv = data_season_tr[i].groupby(['weather','temp'], as_index=False)['humidity'].mean().rename(columns={'humidity':'mean-temp'})
       #data_season_tr[i] = data_season_tr[i].merge(fv, on=['weather','temp'], right_index=True)
       
       #fv = data_season_te[i].groupby(['weather','temp'], as_index=False)['humidity'].mean().rename(columns={'humidity':'mean-temp'})
       #data_season_te[i] = data_season_te[i].merge(fv, on=['weather','temp'], right_index=True)
                     
       #data_season_tr[i]['shift-1'] = data_season_tr[i]['shift-1'] - data_season_te[i]['temp']
       
       
       #groupby(workingday , hour).count.mean()
       
       
       #fv = data_season_tr[i].groupby(['dayofweek'], as_index=False)['humidity'].mean().rename(columns={'humidity':'mean-temp'})
       #data_season_tr[i] = data_season_tr[i].merge(fv, on=['dayofweek']).sort_values('index')
       
       
       
       #data_season_tr[i]['shift-2'] = data_season_tr[i]['atemp'].rolling(6).mean()
       #data_season_te[i]['shift-2'] = data_season_te[i]['atemp'].rolling(6).mean()

       
       
       
       
       
       
       #data_season_tr[i]['shift-2'] = data_season_tr[i]['temp'].shift(3)
       #data_season_te[i]['shift-2'] = data_season_te[i]['temp'].shift(3)
       
       #data_season_tr[i]['shift-2'] = (data_season_tr[i]['shift-2'] + data_season_tr[i]['temp']) / 3
       #data_season_te[i]['shift-2'] = (data_season_te[i]['shift-2'] + data_season_te[i]['temp']) / 3
       
       
       
       #data_season_tr[i]['shift-2'] = data_season_tr[i]['atemp'].shift(1)
       #data_season_te[i]['shift-2'] = data_season_te[i]['atemp'].shift(1)
       
       #data_season_tr[i]['shift-2'] = data_season_tr[i]['shift-2'] - data_season_tr[i]['atemp']
       #data_season_te[i]['shift-2'] = data_season_te[i]['shift-2'] - data_season_te[i]['atemp']
       
       
       
       
       #fv = data_season_tr[i].groupby(['dayofweek'], as_index=False)['humidity'].mean().rename(columns={'humidity':'mean-temp'})
       #data_season_tr[i] = data_season_tr[i].merge(fv, on=['dayofweek']).sort_values('index')
       
       #fv = data_season_te[i].groupby(['dayofweek'], as_index=False)['humidity'].mean().rename(columns={'humidity':'mean-temp'})
       #data_season_te[i] = data_season_te[i].merge(fv, on=['dayofweek']).sort_values('index')
       
       
       #fv = data_season_tr[i].groupby(['weather'], as_index=False)['atemp'].mean().rename(columns={'atemp':'mean-temp'})
       #data_season_tr[i] = data_season_tr[i].merge(fv, on=['weather']).sort_values('index')
       
       #fv = data_season_te[i].groupby(['weather'], as_index=False)['atemp'].mean().rename(columns={'atemp':'mean-temp'})
       #data_season_te[i] = data_season_te[i].merge(fv, on=['weather']).sort_values('index')
       
       
       
       #fv = data_season_te[i].groupby(['weather', 'hour'], as_index=False)['atemp'].mean().rename(columns={'atemp':'mean-temp'})
       #data_season_te[i] = data_season_te[i].merge(fv, on=['weather', 'hour']).sort_values('index')
       
       
       #data_season_tr[i]['diff-tempxatemp'] = data_season_tr[i]['diff-temp'].diff()
       #data_season_te[i]['diff-tempxatemp'] = data_season_te[i]['diff-temp'].diff()
       
       #fv = data_season_tr[i].groupby(['month', 'dayofweek', 'hour'], as_index=False)['temp'].mean().rename(columns={'temp':'mean-temp'})
       #data_season_tr[i] = data_season_tr[i].merge(fv, on=['month', 'dayofweek', 'hour']).sort_values('index')
       
     
       #fv = data_season_tr[i].groupby(['month', 'dayofweek', 'hour'], as_index=False)['temp'].mean().rename(columns={'temp':'mean-temp'})
       #data_season_tr[i] = data_season_tr[i].merge(fv, on=['month', 'dayofweek', 'hour']).sort_values('index')
       
       #fv = data_season_te[i].groupby(['month', 'dayofweek', 'hour'], as_index=False)['temp'].mean().rename(columns={'temp':'mean-temp'})
       #data_season_te[i] = data_season_te[i].merge(fv, on=['month', 'dayofweek', 'hour']).sort_values('index')
       
       
       
       #fv = data_season_tr[i].groupby(['month', 'day'], as_index=False)['temp'].min().rename(columns={'temp':'min-temp'})
       #data_season_tr[i] = data_season_tr[i].merge(fv, on=['month', 'day']).sort_values('index')
       
       #fv = data_season_te[i].groupby(['month', 'day'], as_index=False)['temp'].min().rename(columns={'temp':'min-temp'})
       #data_season_te[i] = data_season_te[i].merge(fv, on=['month', 'day']).sort_values('index')
       
       #fv = data_season_tr[i].groupby(['month', 'day'], as_index=False)['temp'].max().rename(columns={'temp':'max-temp'})
       #data_season_tr[i] = data_season_tr[i].merge(fv, on=['month', 'day']).sort_values('index')
       
       #fv = data_season_te[i].groupby(['month', 'day'], as_index=False)['temp'].max().rename(columns={'temp':'max-temp'})
       #data_season_te[i] = data_season_te[i].merge(fv, on=['month', 'day']).sort_values('index')
       
       
       #fv = data_season_tr[i].groupby(['month', 'hour'], as_index=False)['temp'].mean().rename(columns={'temp':'diff-betwendates'})
       #data_season_tr[i] = data_season_tr[i].merge(fv, on=['month', 'hour']).sort_values('index')
       
       #fv = data_season_te[i].groupby(['month', 'hour'], as_index=False)['temp'].mean().rename(columns={'temp':'diff-betwendates'})
       #data_season_te[i] = data_season_te[i].merge(fv, on=['month', 'hour']).sort_values('index')
       
       
       #fv = data_season_tr[i].groupby(['month', 'hour'], as_index=False)['humidity'].mean().rename(columns={'humidity':'diff-betwendates-humidity'})
       #data_season_tr[i] = data_season_tr[i].merge(fv, on=['month', 'hour']).sort_values('index')
       
       #fv = data_season_te[i].groupby(['month', 'hour'], as_index=False)['humidity'].mean().rename(columns={'humidity':'diff-betwendates-humidity'})
       #data_season_te[i] = data_season_te[i].merge(fv, on=['month', 'hour']).sort_values('index')
       
       #fv = data_season_tr[i].groupby(['month', 'hour'], as_index=False)['atemp'].mean().rename(columns={'atemp':'diff-betweenatemp'})
       #data_season_tr[i] = data_season_tr[i].merge(fv, on=['month', 'hour']).sort_values('index')
       
       #fv = data_season_te[i].groupby(['month', 'hour'], as_index=False)['atemp'].mean().rename(columns={'atemp':'diff-betweenatemp'})
       #data_season_te[i] = data_season_te[i].merge(fv, on=['month', 'hour']).sort_values('index')
       
       #fv = data_season_tr[i].groupby(['month', 'hour'], as_index=False)['windspeed'].mean().rename(columns={'windspeed':'diff-betweenwindspeed'})
       #data_season_tr[i] = data_season_tr[i].merge(fv, on=['month', 'hour']).sort_values('index')
       
       #fv = data_season_te[i].groupby(['month', 'hour'], as_index=False)['windspeed'].mean().rename(columns={'windspeed':'diff-betweenwindspeed'})
       #data_season_te[i] = data_season_te[i].merge(fv, on=['month', 'hour']).sort_values('index')
       
       
       
    return data_season_tr, data_season_te   
       

def get_init():
    
    file_path = 'train.csv'
    file_test = 'test.csv'
    data_train = pd.read_csv(file_path, parse_dates=['datetime'], sep=',')
    data_train['index'] = data_train.index
    data_test = pd.read_csv(file_test, parse_dates=['datetime'], sep=',')
    data_test['index'] = data_test.index
          
    sample_sub = pd.read_csv('sampleSubmission.csv')

    data_train.drop(data_train[data_train.humidity <= 18].index, inplace=True)


    data_train = date_time(data_train)
    data_test = date_time(data_test)

    sample_sub['index'] = sample_sub.index
              
    return data_train, data_test, sample_sub          
  

#######################################################################3

data_train, data_test, sample_sub = get_init()
          
cols = ['holiday','workingday','weather','temp','atemp', 'humidity',
        'windspeed','hour','dayofweek', 'year', 'season'] 
          

data_train['diff-temp'] = data_train['atemp'] - data_train['temp']
data_test['diff-temp'] = data_test['atemp'] - data_test['temp']


#data_train['humidity-diff'] = data_train['humidity'].diff().fillna(0)
#data_test['humidity-diff'] = data_test['humidity'].diff().fillna(0)

          
data_train_less_20 = data_train[data_train['hour'] <= 7]
data_train_less_20['humidity-diff'] = data_train_less_20['humidity'].diff().fillna(0)

fv = data_train_less_20.groupby(['season','hour'])['count'].apply(lambda x: x < 10)
data_train_less_20['lab'] = fv
dc_1 = data_train_less_20.pivot_table(index='hour', columns='lab', aggfunc='size', fill_value=0).apply(lambda r: r/r.sum(), axis=1)
dc_1.reset_index(inplace=True)
data_train_less_20 = data_train_less_20.merge(dc_1, on=['hour'], right_index=True)
data_train_less_20.rename(columns={True: 'Prob_less_10'}, inplace=True)
data_train_less_20.sort_values('index', inplace=True)

#data_test = data_test.merge(dc, on=['hour'], right_index=True)
#data_test.rename(columns={True: 'Prob_less_10'}, inplace=True)
#data_test.sort_values('index', inplace=True)


#data_train_less_20['shift-2'] = data_train_less_20['atemp'].rolling(6).mean().fillna(0)
#data_train_less_20['windspeed-diff'] =data_train_less_20['windspeed'].diff().fillna(0) 
#data_train_less_20['temp'] = data_train_less_20['temp'].astype(int)  
#data_train_less_20['diff-temp'] = data_train_less_20['atemp'] - data_train_less_20['temp']
#data_train_less_20['humidty-temp'] = (data_train_less_20['humidity'] / data_train_less_20['temp']).replace(np.inf, 0)
#data_train_less_20['shift-1'] = data_train_less_20['temp'].shift(1).fillna(0)
#data_train_less_20['shift-1'] = data_train_less_20['shift-1'] - data_train_less_20['temp']
       
data_train_more_20 = data_train[data_train['hour'] > 7]
data_train_more_20['humidity-diff'] = data_train_more_20['humidity'].diff().fillna(0)

fv = data_train_more_20.groupby(['season','hour'])['count'].apply(lambda x: x < 10)
data_train_more_20['lab'] = fv
dc_2 = data_train_more_20.pivot_table(index='hour', columns='lab', aggfunc='size', fill_value=0).apply(lambda r: r/r.sum(), axis=1)
dc_2.reset_index(inplace=True)
data_train_more_20 = data_train_more_20.merge(dc_2, on=['hour'], right_index=True)
data_train_more_20.rename(columns={True: 'Prob_less_10'}, inplace=True)
data_train_more_20.sort_values('index', inplace=True)
#data_train_more_20['shift-2'] = data_train_more_20['atemp'].rolling(6).mean().fillna(0)

#data_train_more_20['windspeed-diff'] =data_train_more_20['windspeed'].diff().fillna(0)   
#data_train_more_20['temp'] = data_train_more_20['temp'].astype(int)  
#data_train_more_20['diff-temp'] = data_train_more_20['atemp'] - data_train_more_20['temp']
#data_train_more_20['humidty-temp'] = (data_train_more_20['humidity'] / data_train_more_20['temp']).replace(np.inf, 0)
#data_train_more_20['shift-1'] = data_train_more_20['temp'].shift(1).fillna(0)
#data_train_more_20['shift-1'] = data_train_more_20['shift-1'] - data_train_more_20['temp']
       
       

data_test_less_20 = data_test[data_test['hour'] <= 7]
data_test_less_20['humidity-diff'] = data_test_less_20['humidity'].diff().fillna(0)

data_test_less_20 = data_test_less_20.merge(dc_1, on=['hour'], right_index=True)
data_test_less_20.rename(columns={True: 'Prob_less_10'}, inplace=True)
data_test_less_20.sort_values('index', inplace=True)

#data_test_less_20['shift-2'] = data_test_less_20['atemp'].rolling(6).mean().fillna(0)


 
#data_test_less_20['windspeed-diff'] = data_test_less_20['windspeed'].diff().fillna(0) 
#data_test_less_20['temp'] = data_test_less_20['temp'].astype(int)  
#data_test_less_20['diff-temp'] = data_test_less_20['atemp'] - data_test_less_20['temp']
#data_test_less_20['humidty-temp'] = (data_test_less_20['humidity'] / data_test_less_20['temp']).replace(np.inf, 0)
#data_test_less_20['shift-1'] = data_test_less_20['temp'].shift(1).fillna(0)
#data_test_less_20['shift-1'] = data_test_less_20['shift-1'] - data_test_less_20['temp']
       


data_test_more_20 = data_test[data_test['hour'] > 7]
data_test_more_20['humidity-diff'] = data_test_more_20['humidity'].diff().fillna(0)
#data_test_more_20['shift-2'] = data_test_more_20['atemp'].rolling(6).mean().fillna(0)

data_test_more_20 = data_test_more_20.merge(dc_2, on=['hour'], right_index=True)
data_test_more_20.rename(columns={True: 'Prob_less_10'}, inplace=True)
data_test_more_20.sort_values('index', inplace=True)

 
#data_test_more_20['windspeed-diff'] = data_test_more_20['windspeed'].diff().fillna(0)  
#data_test_more_20['temp'] = data_test_more_20['temp'].astype(int)  
#data_test_more_20['diff-temp'] = data_test_more_20['atemp'] - data_test_more_20['temp']
#data_test_more_20['humidty-temp'] = (data_test_more_20['humidity'] / data_test_more_20['temp']).replace(np.inf, 0)
#data_test_more_20['shift-1'] = data_test_more_20['temp'].shift(1).fillna(0)
#data_test_more_20['shift-1'] = data_test_more_20['shift-1'] - data_test_more_20['temp']
       


cols.append('humidity-diff')
cols.append('diff-temp')
cols.append('Prob_less_10')
#cols.append('shift-2')
#cols.append('humidty-temp')
#cols.append('shift-1')

#multi_1 = 0.90
          
ml_dt_1 = DecisionTreeRegressor(max_depth=9, random_state = 42)
ml_dt_1.fit(data_train_less_20[cols], data_train_less_20['count'])         
data_fi_1 = ml_dt_1.predict(data_test_less_20[cols])  
data_sea_1 = {'datetime': data_test_less_20['datetime'], 'index': data_test_less_20['index'], 'count': data_fi_1.astype(int) }
data_sea_1 = pd.DataFrame(data=data_sea_1)

ml_dt_2 = DecisionTreeRegressor(max_depth=9, random_state = 42)
ml_dt_2.fit(data_train_more_20[cols], data_train_more_20['count'])   
data_fi_2 = ml_dt_2.predict(data_test_more_20[cols])
data_sea_2 = {'datetime': data_test_more_20['datetime'], 'index': data_test_more_20['index'], 'count': data_fi_2.astype(int) }
data_sea_2 = pd.DataFrame(data=data_sea_2)
  
     
lista_final_sub = pd.concat([data_sea_1,data_sea_2], axis=0)
lista_final_sub.sort_values('index', inplace=True) 
lista_final_sub = lista_final_sub[['datetime', 'count']]
lista_final_sub['count'] = lista_final_sub['count'].astype(int)
#lista_final_sub.sort_index(inplace=True)
lista_final_sub.to_csv('sub.csv', index=False)


######################################################################

data_train, data_test, sample_sub = get_init()

  
####separar por season

data_season_tr = sep_season(data_train)
data_season_te = sep_season(data_test)
    
lista = data_season_tr.keys()
cols = ['holiday','workingday','weather','temp','atemp', 'humidity',
        'windspeed','hour','dayofweek', 'year'] 

cols.append('humidity-diff')
cols.append('diff-temp')
cols.append('humidty-temp')
cols.append('shift-1')
#cols.append('shift-2')
#cols.append('mean-temp')
#cols.append('diff-tempxatemp')
#cols.append('mean-temp')

data_season_tr, data_season_te = feat_eng(data_season_tr, data_season_te, lista)




mean_lr = []
mean_dt = []
mean_rf = []
mean_pred = []

dict_ver = dict()
dict_pred = dict()

for i in lista:
    data_tr = data_season_tr[i].copy()
    data_tr, data_te = get_tr_te(data_tr) 
    
    #ml_rf = RandomForestRegressor(max_depth=3, random_state=42)
    #ml_rf.fit(data_tr[cols], data_tr['count'])     
     
    ml_dt = DecisionTreeRegressor(max_depth=8, random_state = 42)
    ml_dt.fit(data_tr[cols].fillna(0), data_tr['count'])
    #ml_lr = linear_model.LinearRegression()
    #ml_lr.fit(data_tr[cols], data_tr['count'])
    #ml_dt_2 = DecisionTreeRegressor(max_depth=2, random_state = 42)
    #ml_dt_2.fit(data_tr[cols], data_tr['count'])
    
    
    #final_pred = (ml_dt_2.predict(data_te[cols]) + ml_dt.predict(data_te[cols])) / 2
    #met_pred_final = rmsle(data_te['count'],final_pred)
    
    #met_lr = rmsle(data_te['count'],ml_dt_2.predict(data_te[cols]))
    met_dt = rmsle(data_te['count'],ml_dt.predict(data_te[cols].fillna(0)))
    dict_ver[i] = data_te['count'].values
    dict_pred[i] = ml_dt.predict(data_te[cols].fillna(0))
    #met_rf = rmsle(data_te['count'],ml_rf.predict(data_te[cols]))
    #mean_lr.append(met_lr)
    mean_dt.append(met_dt)
    #mean_rf.append(met_rf)
    #mean_pred.append(met_pred_final)
    #print('rmsle-season:{}, {}'.format(i, met_lr))
    print('rmsle-season:{}, {}'.format(i, met_dt))
    
    #importances = ml_dt.feature_importances_
    #indices = np.argsort(importances)[::-1]


    #for f in range(data_tr[cols].shape[1]):
    #    print("%d. feature %d (%f)" % (f, indices[f], importances[indices[f]]))

    #print('rmsle-season-conjunto:{}, {}'.format(i,met_pred_final))




#Treinar com todos os dados
#multi_1 = 0.86
dici_season = dict()   
for i in lista:
    data_tr = data_season_tr[i].fillna(0).copy()
      
    ml_dt = DecisionTreeRegressor(max_depth=9, random_state = 42)
    ml_dt.fit(data_tr[cols], data_tr['count'])
    #ml_lr = linear_model.LinearRegression()
    #ml_lr.fit(data_tr[cols], data_tr['count'])
    dici_season[i] = ml_dt

lista_final = []          
for i in lista:
    data_te = data_season_te[i].fillna(0).copy()
    data_fi = dici_season[i].predict(data_te[cols]) 
    data_sea = {'datetime': data_te['datetime'], 'index': data_te['index'], 'count': data_fi.astype(int) }
    #d = {'col1': [1, 2], 'col2': [3, 4]}
    #df = pd.DataFrame(data=d)
    dat_season = pd.DataFrame(data=data_sea)
    lista_final.append(dat_season)           
               
    
   
lista_final_sub_1 = pd.concat(lista_final, axis=0) 
lista_final_sub_1 = lista_final_sub_1[['datetime', 'count']]
lista_final_sub_1['count'] = lista_final_sub_1['count'].astype(int)
#lista_final_sub_1.reset_index(drop=True, inplace=True)
#lista_final_sub_1.to_csv('sub_1.csv', index=False)



###################################################
data_train, data_test, sample_sub = get_init()


cols = ['holiday','workingday','weather','temp','atemp', 'humidity',
        'windspeed','hour','dayofweek', 'year', 'season'] 

ml_dt = DecisionTreeRegressor(max_depth=13, random_state = 42)
data_train['humidity-diff'] = data_train['humidity'].diff().fillna(0)
data_test['humidity-diff'] = data_test['humidity'].diff().fillna(0)

fv = data_train.groupby(['season','hour'])['count'].apply(lambda x: x < 10)
data_train['lab'] = fv
dc = data_train.pivot_table(index='hour', columns='lab', aggfunc='size', fill_value=0).apply(lambda r: r/r.sum(), axis=1)
dc.reset_index(inplace=True)
data_train = data_train.merge(dc, on=['hour'], right_index=True)
data_train.rename(columns={True: 'Prob_less_10'}, inplace=True)
data_train.sort_values('index', inplace=True)
data_test = data_test.merge(dc, on=['hour'], right_index=True)
data_test.rename(columns={True: 'Prob_less_10'}, inplace=True)
data_test.sort_values('index', inplace=True)

'''fv = data_train.groupby('hour')['count'].apply(lambda x: (x > 10) & (x < 20))
data_train['lab'] = fv
dc = data_train.pivot_table(index='hour', columns='lab', aggfunc='size', fill_value=0).apply(lambda r: r/r.sum(), axis=1)
dc.reset_index(inplace=True)
data_train = data_train.merge(dc, on=['hour'], right_index=True)
data_train.rename(columns={True: 'Prob_lessq_10'}, inplace=True)
data_train.sort_values('index', inplace=True)
data_test = data_test.merge(dc, on=['hour'], right_index=True)
data_test.rename(columns={True: 'Prob_lessq_10'}, inplace=True)
data_test.sort_values('index', inplace=True)'''



cols.append('humidity-diff')
#cols.append('Prob_less_10')
#cols.append('Prob_lessq_10')
#cols.append('diff-temp')
#cols.append('shift-1')

#multi = 0.86

ml_dt.fit(data_train[cols], data_train['count'])
pred = ml_dt.predict(data_test[cols]) 
data_sea = {'datetime': data_test['datetime'], 'index': data_test['index'], 'count': pred.astype(int) }
lista_final_sub_2 = pd.DataFrame(data=data_sea)
lista_final_sub_2[['count', 'datetime']].to_csv('sub.csv', index=False)

######################

lista__final_10 = lista_final_sub.copy()

lista__final_10['count'] = (lista_final_sub['count'] + lista_final_sub_1['count'] + lista_final_sub_2['count']) / 3

lista__final_10.to_csv('sub.csv', index=False)



#####################################################################





importances = ml_dt.feature_importances_
indices = np.argsort(importances)[::-1]


'''for f in range(data_train_final[cols].shape[1]):
    print("%d. feature %d (%f)" % (f, indices[f], importances[indices[f]]))'''



for f, name in enumerate(cols):
     #n, = np.where(indices == f)
     print("%d. feature %s (%f)" % (f, name, importances[f]))

#tree(ml_dt)






plt.hist(data_season_tr[4]['shift-1'].fillna(0), bins=10,label='Treino')
plt.hist(data_season_te[4]['shift-1'].fillna(0), bins=10, label= 'Teste')


#data_season_tr[1][(data_season_tr[1]['day'] == 1) & (data_season_tr[1]['month'] == 1)][['datetime', 'count','temp','atemp','hour']]


#data_season_tr[1][data_season_tr[1]['hour'] == 0][['datetime', 'count','temp','atemp','hour']]


for i in lista:
    lit = np.unique(data_season_tr[i]['hour'])
    for j in lit:
       mean = np.mean(data_season_tr[i][data_season_tr[i]['hour'] == j]['count'])
       std = np.std(data_season_tr[i][data_season_tr[i]['hour'] == j]['count'])
       print('Season:{} - hour:{} - mean:{} - std:{}'.format(i,j,mean, std))


#np.square(np.log(dict_pred[1] + 1) - np.log(dict_ver[1] + 1)).mean() ** 0.5


#data_season_tr[1]['temp'].rolling(2).max().head(n=10)

from sklearn.tree import DecisionTreeClassifier

cols = ['holiday','workingday','weather','temp','atemp', 'humidity',
        'windspeed','hour','dayofweek', 'year'] 


data_train_less_10 = data_train[data_train['count'] <= 10]
data_train_more_10 = data_train[data_train['count'] > 10]


data_train_less_10['label_art'] = 0
data_train_more_10['label_art'] = 1
                  
data_train_final = pd.concat([data_train_less_10, data_train_more_10])                  

dt = DecisionTreeClassifier(max_depth=7, random_state = 42)
dt.fit(data_train_final[cols], data_train_final['label_art'])


importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]


'''for f in range(data_train_final[cols].shape[1]):
    print("%d. feature %d (%f)" % (f, indices[f], importances[indices[f]]))'''



for f, name in enumerate(cols):
     #n, = np.where(indices == f)
     print("%d. feature %s (%f)" % (f, name, importances[f]))



plt.hist(data_season_tr[1]['mean-temp'], bins=10,label='Treino')
plt.hist(data_season_te[1]['mean-temp'], bins=10, label= 'Teste')





xgb_params_2 = {
    'n_estimators': 300,
    'eta': 0.02,
    #'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'silent': True
     
                 }







       
       

