# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:20:23 2017

@author: rebli
"""
from __future__ import division
import pandas as pd
import xgboost as xgb
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV   
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, rand
from scipy import sparse
from sklearn.cluster import KMeans
from string import punctuation
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import rbf_kernel,cosine_similarity,linear_kernel,polynomial_kernel,sigmoid_kernel,laplacian_kernel,chi2_kernel
from datetime import date
from datetime import datetime
from tabulate import tabulate
import funcoes
reload(funcoes)



  
     
lista = ['latitude',
         'longitude'
         ]

df_test = pd.read_json(open("test.json", "r"))
listing_id = df_test['listing_id']
listing_id = listing_id[:,np.newaxis]
df_test = df_test.drop(lista,1)


df = pd.read_json("train.json")
names_columns = df.columns.values
interest_level = df['interest_level']


le = funcoes.Labelencoder(interest_level)
interest_level = le.transform(interest_level)
df = df.drop('interest_level',1)
df = df.drop(lista,1)
df['interest_level'] = interest_level
  

#df['price'] = np.log(df['price'])
#df_test['price'] = np.log(df_test['price'])

df,df_test,mat_1  = funcoes.bat_1(df,df_test)
#df,df_test,mat_20 = bed_1(df,df_test)
#df,df_test,price_0,price_1,price_2,distances = bat_2(df,df_test)
#df,df_test,conj_num,mat_5 = bat_3(df,df_test)
#df,df_test, mat_6, conj_num1 = bat_4(df,df_test)
#df,df_test,mat_9,conj_num2 = bat_5(df,df_test,conj_num)

df['agg_bed_bath'] = df.apply(funcoes.preco_agg,axis=1) 
df_test['agg_bed_bath'] = df_test.apply(funcoes.preco_agg,axis=1) 
df['agg_price_unique'] = df['price'] / df['agg_bed_bath']
df_test['agg_price_unique'] = df_test['price'] / df_test['agg_bed_bath']

                      
            
df = df.drop('agg_bed_bath',1)
df_test = df_test.drop('agg_bed_bath',1)



df['DATE_EXTRACTION'] = pd.to_datetime(df['created'])
df_test['DATE_EXTRACTION'] = pd.to_datetime(df_test['created'])
df["date"] = df.DATE_EXTRACTION.map(lambda x: x.strftime('%Y-%m-%d'))
df_test["date"] = df_test.DATE_EXTRACTION.map(lambda x: x.strftime('%Y-%m-%d'))

df = df.drop('DATE_EXTRACTION',1)
df_test = df_test.drop('DATE_EXTRACTION',1)
df = df.drop('created',1)
df_test = df_test.drop('created',1)

df['month'] = pd.DatetimeIndex(df['date']).month
df_test['month'] = pd.DatetimeIndex(df_test['date']).month

df['week_month'] = df.apply(funcoes.get_week_of_month,axis=1)
df_test['week_month'] = df_test.apply(funcoes.get_week_of_month,axis=1)

df = df.drop('date',1)       
df_test = df_test.drop('date',1)   


df["num_photos"] = df["photos"].apply(len)
df_test["num_photos"] = df_test["photos"].apply(len)

# count of "features" #
df["num_features"] = df["features"].apply(len)
df_test["num_features"] = df_test["features"].apply(len)

# count of words present in description column #
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df_test["num_description_words"] = df_test["description"].apply(lambda x: len(x.split(" ")))






#df["price_t"] = df["price"]/df["bedrooms"] 
#df_test["price_t"] = df_test["price"]/df_test["bedrooms"] 
#df["room_sum"] = df["bedrooms"]+df["bathrooms"] 
df["room_dif"] = df["bedrooms"]-df["bathrooms"] 
#df["price_t1"] = df["price"]/df["room_sum"]
#df["fold_t1"] = df["bedrooms"]/df["room_sum"]

df_test["room_dif"] = df_test["bedrooms"]-df_test["bathrooms"] 
#df_test["room_sum"] = df_test["bedrooms"]+df_test["bathrooms"] 
#df_test["price_t1"] = df_test["price"]/df_test["room_sum"]
#df_test["fold_t1"] = df_test["bedrooms"]/df_test["room_sum"]






df['price_per_bath'] = df.apply(funcoes.preco_agg_bath,axis=1) 
df_test['price_per_bath'] = df_test.apply(funcoes.preco_agg_bath,axis=1)
df['price_per_bed'] = df.apply(funcoes.preco_agg_bed,axis=1) 
df_test['price_per_bed'] = df_test.apply(funcoes.preco_agg_bed,axis=1)
df['diff_bed_bath'] = df['price_per_bath'] - df['price_per_bed']
df_test['diff_bed_bath'] = df_test['price_per_bath'] - df_test['price_per_bed']


df,df_test = funcoes.contar_dias_mes(df,df_test)

df,df_test = funcoes.bins(df,df_test)


#primeiro modelo contendo todas as variaveis relevantes
features_to_use  = ["bathrooms", "bedrooms", "price"]

features_to_use.extend(["prob_0","agg_price_unique",'month','week_month','num_photos','num_features',
                        'num_description_words','room_dif','diff_bed_bath','listing_id','bins','bins3',
                        'price_per_bath','contando',
                        'price_per_bed','prob_10','prob_11','prob_22'])  



features_to_use1 = ["bathrooms", "bedrooms", "price"]
features_to_use1.extend(["prob_0",'month','week_month','num_photos','num_features',
                        'num_description_words','room_dif','listing_id']) 


features_to_use2 = ["bathrooms", "bedrooms", "price"]
features_to_use2.extend(["agg_price_unique",'month','week_month','num_photos','num_features',
                        'num_description_words','room_dif']) 


features_to_use4  = ["bathrooms", "bedrooms", "price"]
features_to_use4.extend(["display_address", "manager_id", "building_id", "street_address"]) 


features_to_use5 = []
features_to_use5.extend(["display_address", "manager_id", "building_id", "street_address"]) 

features_to_use6  = ["bathrooms", "bedrooms", "price"]
features_to_use6.extend(["display_address", "manager_id", "street_address",'num_photos','room_dif']) 


features_to_use7  = ["bathrooms", "bedrooms", "price"]
features_to_use7.extend(["display_address", "manager_id",'agg_price_unique','week_month']) 

features_to_use8  = ["bathrooms", "bedrooms", "price"]
features_to_use8.extend(["agg_price_unique",'month','week_month','num_photos','num_features']) 


   
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[f].values) + list(df_test[f].values))
            df[f] = lbl.transform(list(df[f].values))
            df_test[f] = lbl.transform(list(df_test[f].values))
            features_to_use.append(f)
            features_to_use1.append(f)
            features_to_use2.append(f)
            


features_to_use1.pop(14)
features_to_use1.pop(13)


features_to_use3 = list(features_to_use)



'''df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

df_test = df_test.replace([np.inf, -np.inf], np.nan)
df_test = df_test.dropna()


interest_level = df['interest_level']'''


#valores_treino,valores_teste,total_teste,tt,probabilidades,df,df_test = funcoes.managertudo(df,df_test)
valores_treino,valores_teste,total_teste,tt,probabilidades,df,df_test = funcoes.managertudodois(df,df_test)
#df,df_test,mat_10 = manager(df,df_test)
#df,df_test,mat_10 = manager1(df,df_test)
#df,df_test,mat_10 = manager2(df,df_test)

'''valores_building = pd.value_counts(df['building_id']).index
valores_building2 = pd.value_counts(df['building_id'])
                   
valores_display = pd.value_counts(df['display_address']).index
valores_display2 = pd.value_counts(df['display_address'])



valores_manager = pd.value_counts(df['manager_id']).index
valores_manager2 = pd.value_counts(df['manager_id'])
                                 
valores_street = pd.value_counts(df['street_address']).index
valores_street2 = pd.value_counts(df['street_address'])


                                

for j in range(0,len(valores_building)):
       df.loc[df['building_id'] == valores_building[j] ,['building_id']] = valores_building2[valores_building[j]]                        



for j in range(0,len(valores_display)):
       df.loc[df['display_address'] == valores_display[j] ,['display_address']] = valores_display2[valores_display[j]]
       
       
for j in range(0,len(valores_manager)):
       df.loc[df['manager_id'] == valores_manager[j] ,['manager_id']] = valores_manager2[valores_manager[j]]



for j in range(0,len(valores_street)):
       df.loc[df['street_address'] == valores_street[j] ,['street_address']] = valores_street2[valores_street[j]]'''


       
'''df['interest_level'] = interest_level
df = df[df['month'] == 6] 

interest_level = df['interest_level']'''

                   
'''f1 = df[df['interest_level'] == 1]
f2 = df[df['interest_level'] == 2]
f3 = df[df['interest_level'] == 0]

f1['description_clean'] = f1['description'].apply(funcoes.remove_punctuation)
f2['description_clean'] = f2['description'].apply(funcoes.remove_punctuation)                  
f3['description_clean'] = f3['description'].apply(funcoes.remove_punctuation)                  

vectorizer = CountVectorizer(stop_words='english',max_features=20)
vectorizer1 = CountVectorizer(stop_words='english',max_features=20)
vectorizer2 = CountVectorizer(stop_words='english',max_features=20)

train_matrix1 = vectorizer.fit_transform(f1["description_clean"]).toarray()
train_matrix2 = vectorizer1.fit_transform(f2["description_clean"]).toarray()
train_matrix3 = vectorizer2.fit_transform(f3["description_clean"]).toarray()


vocab = vectorizer.get_feature_names()
vocab1 = vectorizer1.get_feature_names()
vocab2 = vectorizer2.get_feature_names()'''

                  

df['description_clean'] = df['description'].apply(funcoes.remove_punctuation)
df_test['description_clean'] = df_test['description'].apply(funcoes.remove_punctuation)
df['features'] = df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
df_test['features'] = df_test["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
vectorizer = CountVectorizer(stop_words='english',max_features=100)

train_matrix2 = vectorizer.fit_transform(df["features"])
train_matrix1 = vectorizer.fit_transform(df["description_clean"])
dist_train = train_matrix1.toarray()
dist_train2 = train_matrix2.toarray()

dist_train_total = dist_train + dist_train2
dist_train_total = pd.DataFrame(dist_train_total[:,0:25])
dist_train_total_ir = dist_train_total.copy()
dist_train_total_ir['interest_level'] = interest_level
  

test_matrix2 = vectorizer.fit_transform(df_test["features"])
test_matrix1 = vectorizer.fit_transform(df_test["description_clean"])
dist_test = test_matrix1.toarray()
dist_test2 = test_matrix2.toarray()

dist_teste_total = dist_test + dist_test2
dist_teste_total = pd.DataFrame(dist_teste_total[:,0:25]).copy()
dist_teste_total_ir = dist_teste_total.copy()

#valores_treino,valores_teste,total_teste,tt,probabilidades,merda,merda1 = funcoes.managertudoquarto(dist_train_total_ir,dist_teste_total_ir)


'''liste = range(10,450,10)
for i in enumerate(liste):
    
    vectorizer = CountVectorizer(stop_words='english',max_features=i[1])

    train_matrix1 = vectorizer.fit_transform(df["features"]).toarray()
    #train_matrix2 = vectorizer.fit_transform(df["description_clean"]).toarray()
    vocab = vectorizer.get_feature_names()
    #dist = np.sum(train_matrix1, axis=0)

    vectorizer1 = CountVectorizer(stop_words='english',max_features=i[1])
    
    test_matrix11 = vectorizer1.fit_transform(df_test['features']).toarray()
    #test_matrix12 = vectorizer1.fit_transform(df_test["description_clean"]).toarray()
    vocab1 = vectorizer1.get_feature_names()
    #dist1 = np.sum(train_matrix1, axis=0)
    vocab = set(vocab)
    vocab1 = set(vocab1)
    intersection_count = len(vocab & vocab1)
    print(intersection_count)
    print(intersection_count == len(vocab1))'''




            
'''dados = df[['num_photos','num_features']] 
dados['interest_level'] = interest_level'''
     
'''clusterer = KMeans(n_clusters=3, init='k-means++', random_state=10)
cluster_labels = clusterer.fit_predict(dados)'''
'''valor_train = rbf_kernel(dist_train,clusterer.cluster_centers_)
valor_test = rbf_kernel(dist_test,clusterer.cluster_centers_)'''
  



'''df['cluster_rbf_0'] = np.max(valor_train,1)
#df['cluster_rbf_1'] = valor_train[:,1]
#df['cluster_rbf_2'] = valor_train[:,2]
  
df_test['cluster_rbf_0'] = np.max(valor_test,1)
#df_test['cluster_rbf_1'] = valor_test[:,1]
#df_test['cluster_rbf_2'] = valor_test[:,2]
            

valor_train_cosine = cosine_similarity(dist_train,clusterer.cluster_centers_)
valor_train_linear = linear_kernel(dist_train,clusterer.cluster_centers_)
valor_train_poly = polynomial_kernel(dist_train,clusterer.cluster_centers_)
valor_train_sigmoid = sigmoid_kernel(dist_train,clusterer.cluster_centers_)
valor_train_laplace = laplacian_kernel(dist_train,clusterer.cluster_centers_)
valor_train_chi2 = chi2_kernel(dist_train,clusterer.cluster_centers_)

valor_test_cosine = cosine_similarity(dist_test,clusterer.cluster_centers_)
valor_test_linear = linear_kernel(dist_test,clusterer.cluster_centers_)
valor_test_poly = polynomial_kernel(dist_test,clusterer.cluster_centers_)
valor_test_sigmoid = sigmoid_kernel(dist_test,clusterer.cluster_centers_)
valor_test_laplace = laplacian_kernel(dist_test,clusterer.cluster_centers_)
valor_test_chi2 = chi2_kernel(dist_test,clusterer.cluster_centers_)
 



df['cluster_0'] = np.max(valor_train_cosine,1)
#df['cluster_1'] = valor_train_cosine[:,1]
#df['cluster_2'] = valor_train_cosine[:,2]


df['cluster_00'] = np.max(valor_train_linear,1)
#df['cluster_11'] = valor_train_linear[:,1]
#df['cluster_22'] = valor_train_linear[:,2]


df['cluster_000'] = np.max(valor_train_poly,1)
#df['cluster_111'] = valor_train_poly [:,1]
#df['cluster_222'] = valor_train_poly [:,2]


df['cluster_0000'] = np.max(valor_train_sigmoid,1)
#df['cluster_1111'] = valor_train_sigmoid[:,1]
#df['cluster_2222'] = valor_train_sigmoid[:,2]

df['cluster_00000'] = np.max(valor_train_laplace,1)
#df['cluster_11111'] = valor_train_laplace[:,1]
#df['cluster_22222'] = valor_train_laplace[:,2]



df['cluster_000000'] = np.max( valor_train_chi2,1)
#df['cluster_111111'] = valor_train_chi2[:,1]
#df['cluster_222222'] = valor_train_chi2[:,2]




df_test['cluster_0'] = np.max(valor_test,1)
#df_test['cluster_1'] = valor_test[:,1]
#df_test['cluster_2'] = valor_test[:,2]



df_test['cluster_00'] = np.max(valor_test_linear,1)
#df_test['cluster_11'] = valor_test_linear[:,1]
#df_test['cluster_22'] = valor_test_linear[:,2]


df_test['cluster_000'] = np.max(valor_test_poly,1)
#df_test['cluster_111'] = valor_test_poly [:,1]
#df_test['cluster_222'] = valor_test_poly [:,2]


df_test['cluster_0000'] = np.max(valor_test_sigmoid,1)
#df_test['cluster_1111'] = valor_test_sigmoid[:,1]
#df_test['cluster_2222'] = valor_test_sigmoid[:,2]


df_test['cluster_00000'] = np.max(valor_test_laplace,1)
#df_test['cluster_11111'] = valor_test_laplace[:,1]
#df_test['cluster_22222'] = valor_test_laplace[:,2]



df_test['cluster_000000'] = np.max(valor_test_chi2,1)
#df_test['cluster_111111'] = valor_test_chi2[:,1]
#df_test['cluster_222222'] = valor_test_chi2[:,2]'''

        
        
        
#features_to_use_model = list(features_to_use)  
#features_use_1 = list(features_to_use)
 

#feat = list(features_to_use)        

train_X1 = sparse.hstack([df[features_to_use],train_matrix2,train_matrix1]).tocsr()
test_X1 = sparse.hstack([df_test[features_to_use],test_matrix2,test_matrix1]).tocsr()


train_X2 = sparse.hstack([df[features_to_use1],train_matrix2,train_matrix1]).tocsr()
test_X2 = sparse.hstack([df_test[features_to_use1],test_matrix2,test_matrix1]).tocsr() 


train_X3 = sparse.hstack([df[features_to_use2],train_matrix2,train_matrix1]).tocsr()
test_X3 = sparse.hstack([df_test[features_to_use2],test_matrix2,test_matrix1]).tocsr() 

train_X4 = sparse.hstack([df[features_to_use3],train_matrix2,train_matrix1]).tocsr()
test_X4 = sparse.hstack([df_test[features_to_use3],test_matrix2,test_matrix1]).tocsr()

train_X5 = sparse.hstack([df[features_to_use4],train_matrix2,train_matrix1]).tocsr()
test_X5 = sparse.hstack([df_test[features_to_use4],test_matrix2,test_matrix1]).tocsr()

train_X6 = sparse.hstack([df[features_to_use5],train_matrix2,train_matrix1]).tocsr()
test_X6 = sparse.hstack([df_test[features_to_use5],test_matrix2,test_matrix1]).tocsr() 

train_X7 = sparse.hstack([df[features_to_use6],train_matrix2,train_matrix1]).tocsr()
test_X7 = sparse.hstack([df_test[features_to_use6],test_matrix2,test_matrix1]).tocsr() 

train_X8 = sparse.hstack([df[features_to_use7],train_matrix2,train_matrix1]).tocsr()
test_X8 = sparse.hstack([df_test[features_to_use7],test_matrix2,test_matrix1]).tocsr() 

train_X9 = sparse.hstack([df[features_to_use8],train_matrix2,train_matrix1]).tocsr()
test_X9 = sparse.hstack([df_test[features_to_use8],test_matrix2,test_matrix1]).tocsr() 

  

df = df.drop(['photos','features','description','price_per_bath','price_per_bed'],1)
df_test = df_test.drop(['photos','features','description','price_per_bath','price_per_bed','description_clean'],1)





'''train_X1 = sparse.hstack([df[features_to_use], train_matrix1]).tocsr()
test_X1 = sparse.hstack([df_test[features_to_use], test_matrix1]).tocsr() '''                     


'''train_X1 = scipy.sparse.csr_matrix(df[features_to_use])
test_X1 = scipy.sparse.csr_matrix(df_test[features_to_use])'''
                     


'''target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)'''


                            
df = df.drop('interest_level',1)
                                
X_train, X_test, y_train, y_test = train_test_split(train_X1, interest_level, test_size=0.20, random_state=7)
predictors = df.columns.values
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 silent=1,
 min_child_weight=7,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
agb = funcoes.modelfit(xgb1, X_train,y_train)
print(log_loss(y_test,agb.predict_proba(X_test)))



X_train, X_test, y_train, y_test = train_test_split(train_X2, interest_level, test_size=0.20, random_state=7)
predictors = df.columns.values
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 silent=1,
 min_child_weight=7,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
agb1 = modelfit(xgb1, X_train,y_train)
print(log_loss(y_test,agb1.predict_proba(X_test)))



X_train, X_test, y_train, y_test = train_test_split(train_X3, interest_level, test_size=0.20, random_state=7)
predictors = df.columns.values
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 silent=1,
 min_child_weight=7,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
agb2 = funcoes.modelfit(xgb1, X_train,y_train)
print(log_loss(y_test,agb2.predict_proba(X_test)))

X_train, X_test, y_train, y_test = train_test_split(train_X4, interest_level, test_size=0.20, random_state=7)
predictors = df.columns.values
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 silent=1,
 min_child_weight=7,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
agb3 = funcoes.modelfit(xgb1, X_train,y_train)
print(log_loss(y_test,agb3.predict_proba(X_test)))

X_train, X_test, y_train, y_test = train_test_split(train_X5, interest_level, test_size=0.20, random_state=7)
predictors = df.columns.values
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 silent=1,
 min_child_weight=7,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
agb4 = modelfit(xgb1, X_train,y_train)
print(log_loss(y_test,agb4.predict_proba(X_test)))



X_train, X_test, y_train, y_test = train_test_split(train_X6, interest_level, test_size=0.20, random_state=7)
predictors = df.columns.values
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 silent=1,
 min_child_weight=7,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
agb5 = modelfit(xgb1, X_train,y_train)
print(log_loss(y_test,agb5.predict_proba(X_test)))


X_train, X_test, y_train, y_test = train_test_split(train_X7, interest_level, test_size=0.20, random_state=7)
predictors = df.columns.values
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 silent=1,
 min_child_weight=7,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
agb6 = modelfit(xgb1, X_train,y_train)
print(log_loss(y_test,agb6.predict_proba(X_test)))



X_train, X_test, y_train, y_test = train_test_split(train_X8, interest_level, test_size=0.20, random_state=7)
predictors = df.columns.values
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 silent=1,
 min_child_weight=7,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
agb7 = modelfit(xgb1, X_train,y_train)
print(log_loss(y_test,agb7.predict_proba(X_test)))


X_train, X_test, y_train, y_test = train_test_split(train_X9, interest_level, test_size=0.20, random_state=7)
predictors = df.columns.values
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 silent=1,
 min_child_weight=7,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
agb8 = modelfit(xgb1, X_train,y_train)
print(log_loss(y_test,agb8.predict_proba(X_test)))





param_test1 = {
 'max_depth':range(1,14,2),
 'min_child_weight':range(1,9,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='neg_log_loss',n_jobs=1,iid=False, cv=10)
gsearch1.fit(X_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


'''param_test2 = {
 'max_depth':[8,9,10],
 'min_child_weight':[2,3,4]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=gsearch1.best_params_['max_depth'],
 min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='accuracy',n_jobs=1,iid=False, cv=5)
gsearch2.fit(X_train,y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_'''

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=gsearch1.best_params_['max_depth'],
 min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='neg_log_loss',n_jobs=1,iid=False, cv=10)
gsearch3.fit(X_train,y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


param_test4 = {
 'subsample':[i/10.0 for i in range(6,11)],
 'colsample_bytree':[i/10.0 for i in range(6,11)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177,max_depth=gsearch1.best_params_['max_depth'],
 min_child_weight=gsearch1.best_params_['min_child_weight'], gamma= gsearch3.best_params_['gamma'], subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='neg_log_loss',n_jobs=1,iid=False, cv=10)
gsearch4.fit(X_train,y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


param_test5 = {
 'subsample':[i/100.0 for i in range(75,105,5)],
 'colsample_bytree':[i/100.0 for i in range(75,105,5)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177,max_depth=gsearch1.best_params_['max_depth'],
 min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=gsearch3.best_params_['gamma'], subsample=gsearch4.best_params_['subsample'], 
                                       colsample_bytree=gsearch4.best_params_['colsample_bytree'],
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test5, scoring='neg_log_loss',n_jobs=1,iid=False, cv=10)
gsearch5.fit(X_train,y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177,max_depth=gsearch1.best_params_['max_depth'],
 min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=gsearch3.best_params_['gamma'], subsample=gsearch4.best_params_['subsample'], 
                                       colsample_bytree=gsearch4.best_params_['colsample_bytree'],
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='neg_log_loss',n_jobs=1,iid=False, cv=10)
gsearch6.fit(X_train,y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


param_test7 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177,max_depth=gsearch1.best_params_['max_depth'],
 min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=gsearch3.best_params_['gamma'], subsample=gsearch4.best_params_['subsample'], 
                                       colsample_bytree=gsearch4.best_params_['colsample_bytree'],
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test7, scoring='neg_log_loss',n_jobs=1,iid=False, cv=10)
gsearch7.fit(X_train,y_train)
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_


xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=gsearch1.best_params_['max_depth'],
 min_child_weight=gsearch1.best_params_['min_child_weight'],
 gamma=gsearch3.best_params_['gamma'],
 subsample=gsearch4.best_params_['subsample'],
 colsample_bytree=gsearch4.best_params_['colsample_bytree'],
 reg_alpha=gsearch7.best_params_['reg_alpha'],
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
agl3 = modelfit(xgb3, X_train, y_train)


df = pd.concat([X_train,X_test])
inter = np.concatenate([y_train,y_test]) 
import Principal
reload(Principal)
total = Principal.iniciar(train_X1,interest_level,xgb1)
teste1 = np.asarray(total)



# testeX1_soma = test_X1

primeiro = agb.predict_proba(test_X1)
segundo = agb1.predict_proba(test_X2)
terceiro = agb2.predict_proba(test_X3)
quarto =  agb3.predict_proba(test_X4)
quinto =  agb4.predict_proba(test_X5)
seis = agb5.predict_proba(test_X6)
sete = agb6.predict_proba(test_X7)
oito = agb7.predict_proba(test_X8)


sub = pd.read_csv("sample_submission.csv")
sub['listing_id'] = listing_id
sub.ix[:,1] = primeiro[:,0]
sub.ix[:,2] = primeiro[:,2]
sub.ix[:,3] = primeiro[:,1]   
   
   
sub1 = pd.read_csv("sample_submission.csv")
sub1['listing_id'] = listing_id
sub1.ix[:,1] = segundo[:,0]
sub1.ix[:,2] = segundo[:,2]
sub1.ix[:,3] = segundo[:,1]   


sub2 = pd.read_csv("sample_submission.csv")
sub2['listing_id'] = listing_id
sub2.ix[:,1] = terceiro[:,0]
sub2.ix[:,2] = terceiro[:,2]
sub2.ix[:,3] = terceiro[:,1]  

sub3 = pd.read_csv("sample_submission.csv")
sub3['listing_id'] = listing_id
sub3.ix[:,1] = quarto[:,0]
sub3.ix[:,2] = quarto[:,2]
sub3.ix[:,3] = quarto[:,1]

sub5 = pd.read_csv("sample_submission.csv")
sub5['listing_id'] = listing_id
sub5.ix[:,1] = quinto[:,0]
sub5.ix[:,2] = quinto[:,2]
sub5.ix[:,3] = quinto[:,1]

sub6 = pd.read_csv("sample_submission.csv")
sub6['listing_id'] = listing_id
sub6.ix[:,1] = sete[:,0]
sub6.ix[:,2] = sete[:,2]
sub6.ix[:,3] = sete[:,1]


sub7 = pd.read_csv("sample_submission.csv")
sub7['listing_id'] = listing_id
sub7.ix[:,1] = oito[:,0]
sub7.ix[:,2] = oito[:,2]
sub7.ix[:,3] = oito[:,1]


sub8 = pd.read_csv("sample_submission.csv")
sub8['listing_id'] = listing_id
sub8.ix[:,1] = seis[:,0]
sub8.ix[:,2] = seis[:,2]
sub8.ix[:,3] = seis[:,1]

#sub3.iloc[:,1:4] = (sub.iloc[:,1:4] + sub1.iloc[:,1:4] + sub2.iloc[:,1:4]) / 3
sub4 = pd.read_csv("sample_submission.csv")
sub4['listing_id'] = listing_id
    

sub4.iloc[:,1:4] = (sub.iloc[:,1:4] + sub1.iloc[:,1:4] + sub2.iloc[:,1:4] + sub3.iloc[:,1:4] + sub5.iloc[:,1:4] 
 + sub6.iloc[:,1:4] + sub7.iloc[:,1:4] + sub8.iloc[:,1:4]) / 8


sub4.iloc[:,1:4] = (sub.iloc[:,1:4] + sub3.iloc[:,1:4]) / 2
subb = sub.copy()
sub33 = sub3.copy()

maximo = []
maximo1 = []
maximo2 = []
maximo3 = []
maximo4 = []
maximo5 = []


for i in range(0,len(sub)):
    maximo.append(np.max(sub.iloc[i,1:4]))
    
    
for i in range(0,len(sub1)):
    maximo1.append(np.argmax(sub1.iloc[i,1:4]))


for i in range(0,len(sub2)):
    maximo2.append(np.argmax(sub2.iloc[i,1:4]))    

    
for i in range(0,len(sub3)):
    maximo3.append(np.argmax(sub3.iloc[i,1:4]))     
    

for i in range(0,len(sub4)):
    maximo5.append(np.argmax(sub4.iloc[i,1:4]))  



#pd.value_counts(np.asarray(maximo3) == np.asarray(maximo4))

#sub.to_csv("samples.csv", index=False)
          

soma = np.log(0.75) * 23500
soma1 = np.log(0.125) * 2250
soma2 = np.log(0.125) * 2250

total = soma + soma1 + soma2             
total / -28000
  


'''for i in range(0,len(valores_treino)):
    gg = valores_treino[i]
    print(gg.keys()[0])'''
    
       
   






'''import itertools
tt = list(itertools.product([0,1], repeat=18))
ty = np.asarray(tt)

app = []
for i in range(1,len(ty)):
    ind, = np.where(ty[i,:] == 1)
    if len(ind) <= 7:
        app.append(i)
        

ty = np.delete(ty,app,0)


valores = []
for i in range(1,len(ty[0:400])):
    print(i)
    ind, = np.where(ty[i,:] == 1)
    base_train = X_train.iloc[:,ind[:]]
    base_test = X_test.iloc[:,ind[:]]
    agl3 = modelfit(xgb3, base_train, y_train)
    pred = agl3.predict_proba(base_test)
    tu = log_loss(y_test, pred)
    print "Accuracy Teste: %.4g" % tu
    valores.append(tu)



ind, = np.where(ty[286,:] == 1)
base_train = X_train.iloc[:,ind[:]]
base_test = X_test.iloc[:,ind[:]]
agl3 = modelfit(xgb3, base_train, y_train)
pred = agl3.predict_proba(base_test)
tu = log_loss(y_test, pred)
print "Accuracy Teste: %.4g" % tu


base_testando = df_test.iloc[:,ind[:]]

teste = agl3.predict_proba(base_testando)
sub = pd.read_csv("sample_submission.csv")


sub.ix[:,1] = teste[:,0]
sub.ix[:,2] = teste[:,2]
sub.ix[:,3] = teste[:,1]

sub.to_csv("sample.csv", index=False)'''

 
          
'''lista = []
          
for i in range(0,len(teste[19,:])-1):
               
     if teste[1,i] <= 0.6:
        lista.append(i)


base_train = X_train.drop(X_train.columns[[lista]],1)
base_test = X_test.drop(X_test.columns[[lista]],1)
agl3 = modelfit(xgb3, base_train, y_train)
pred = agl3.predict_proba(base_test)
tu = log_loss(y_test, pred)
print "Accuracy Teste: %.4g" % tu


base_testando = df_test.drop(df_test.columns[[lista]],1)
teste = agl3.predict_proba(base_testando)
sub = pd.read_csv("sample_submission.csv")


sub.ix[:,1] = teste[:,0]
sub.ix[:,2] = teste[:,2]
sub.ix[:,3] = teste[:,1]

sub.to_csv("sample.csv", index=False)'''


'''test_ids = df_test.manager_id.unique()
train_ids = df.manager_id.unique()
intersection_count = len(test_ids & train_ids)
intersection_count == len(test_ids)'''


'''for i in range(n):
     print(i)
     for j in range(n):
         x, y = dist[i, :], dist[j, :]
         dist1[i, j] = np.sqrt(np.sum((x - y)**2))'''


'''mapsize = [33,33] # 5*np.sqrt(len(dados_con))
som = sompy.SOMFactory.build(np.asarray(df[['bathrooms','bedrooms','price']]), mapsize, mask=None, mapshape='planar', lattice='rect', 
                             normalization='var', initialization='random', neighborhood='gaussian', training='batch', name='sompy')
som.train(n_job=1, verbose='info')'''

'''som.component_names = ['1','2','3']
v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)  
v.show(som, what='codebook', which_dim='all', cmap='jet', col_sz=6)''' 

# c = sompy.mapview.View2DPacked()
'''v = sompy.mapview.View2DPacked(2, 2, 'test',text_size=8)  
#first you can do clustering. Currently only K-means on top of the trained som
cl = som.cluster(n_clusters=3)
h = sompy.hitmap.HitMapView(20, 20, 'hitmap', text_size=8, show_text=True)
h.show(som)
# print cl
grid_labels = som.cluster_labels
data_projection = som.project_data(dados_con)
data_clean = np.zeros(shape=(1,len(data_projection)))
data_clean = np.concatenate(data_clean)
for i in range(0,len(data_projection)):
    
    data_clean[i] = grid_labels[data_projection[i]]'''
    
    
from sklearn.decomposition import PCA
from itertools import combinations
lista = list(combinations(range(0,df.iloc[:,0:16].shape[1]),4))
valores = df.columns.values
df1 = df.copy()

df1['interest_level'] = interest_level
  
df1 = df1.replace([np.inf, -np.inf], np.nan) 
df1 = df1.dropna()
interest_level1 = df1['interest_level']
df1 = df1.drop('interest_level',1)
for i in enumerate(lista):
    
    #n_components = n_row * n_col
    estimator = PCA(n_components=3)
    dados = df1.iloc[:,i[1]]
    X_pca = estimator.fit_transform(dados)
    
    fig = plt.figure(1, figsize=(8, 8))
    plt.clf()
    ax = Axes3D(fig) 
    plt.cla()
    ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2],c=interest_level1.astype(np.float))
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    name = "saida2/" + str(i[1]) + ".png"    
    plt.savefig(name)
   
        
    #plt.legend(digits.target_names)
    #plt.xlabel('First Principal Component')
    #plt.ylabel('Second Principal Component') 
    '''fig = plt.figure(1, figsize=(8, 8))
    plt.clf()
    ax = Axes3D(fig) 
    plt.cla()'''
    '''dados = df[[str(valores[i[1][0]]),str(valores[i[1][1]]),str(valores[i[1][2]])]]
    dados = dados.replace(['inf'],0)
    dados = dados.replace(['NaN'],0)'''
    '''dados['interest_level'] = interest_level'''
    '''print(str(valores[i[1][0]]) + "-" + str(valores[i[1][1]]) + '-' + str(valores[i[1][2]]))
    
    clusterer = KMeans(n_clusters=3, init='k-means++', random_state=10)
    cluster_labels = clusterer.fit_predict(dados)
    print(pd.value_counts(cluster_labels))'''
    
    '''ax.scatter(df.iloc[:,i[1][0]], df.iloc[:,i[1][1]],df.iloc[:,i[1][2]],c=interest_level.astype(np.float)   )
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])'''
    #ax.set_xlabel('Malha de controle da Pressao da Zona de queima - PV')
    #ax.set_ylabel('Malha de controle da Pressao da Zona de queima - MV')
    #ax.set_zlabel('Malha de controle da Pressao da Zona de queima - SP')
    '''name = "saida/" + str(valores[i[1][0]]) + "-" + str(valores[i[1][1]]) + '-' + str(valores[i[1][2]]) + ".png"
   '''
    





 
#df.groupby([pd.cut(df.building_id, cont),'street_address', 'manager_id']).bathrooms.count().groupby(level='bathrooms').count()
#tk = pd.cut(dados.price, cont)
'''tp1 = df.groupby(['building_id','bathrooms', 'bathrooms']).building_id.count().keys()
soma = 0
soma1 = 0
for i in range(0,len(tp1)):
    #print(i)  
    valor = df[(df['building_id'] == tp1[i][0]) & (df['street_address'] == tp1[i][1]) & 
    (df['manager_id'] == tp1[i][2])]
    if pd.value_counts(valor['interest_level']).shape[0] == 3:
       soma = soma + 1 
       #print(pd.value_counts(valor['interest_level']))  
    
    if pd.value_counts(valor['interest_level']).shape[0] == 2:
       soma1 = soma1 + 1
    
    
valor_train = set(np.unique(df['room_dif']))
print(len(valor_train))
valor_test = set(np.unique(df_test['room_dif']))
print(len(valor_test))


len(valor_train & valor_test)'''






#fu = pd.value_counts(interest_level)
#fu_keys = fu.keys()
#84

'''df_cop = df.iloc[:,25:190].copy()
df_test_cop = df_test.iloc[:,24:189].copy()

df_cop = merda.iloc[:,26::]
df_test_cop = merda1.iloc[:,25::]'''


df_treino = df.iloc[:,28::].copy()
df_teste =  df_test.iloc[:,27::].copy()


                    
df_cop = df.iloc[:,28::].copy()
df_test_cop = df_test.iloc[:,27::].copy()

df_cop = pd.read_csv("bases/treino.csv")
df_test_cop = pd.read_csv("bases/teste.csv")

for i in range(27,195):
    tg = pd.value_counts(df_test.iloc[:,i])
    print(tg['NaN'])
    
#df_cop['soma'] = 0
agora = []   

definitivo = []
for i in range(0,int((df_cop.shape[1] / 3))):
  print(i)  
  nome = 'pob_0.combination' + str(i)
  nome1 = 'pob_1.combination' + str(i)
  nome2 = 'pob_2.combination' + str(i)
  teste = df_cop.iloc[:][[nome,nome1,nome2]]
  teste = np.asarray(teste)
  temp = []
  for j in range(0,len(teste)):
      temp.append(np.argmax(teste[j]))
  definitivo.append(temp)

dat = pd.DataFrame(definitivo)
dat = dat.T

 
num = 0
numero = []
nomes = []
for i in range(0,len(tt)):
    #name = str(i)
#    print(tt[i])
    tg = pd.value_counts(dat[i] == interest_level)[False]
    perc_tg = tg / len(dat)
    if perc_tg < 0.15:
       numero.append(num)
       nomes.append(tt[i])
    num = num + 1   
    #print("treino:")
    #print(valores_treino[i])
    #print("teste:")
    #print(valores_teste[i])
    #print("soma_total:")
    #print(sum(total_teste[i]))

numero1 = range(55)
excluir=list(set(numero1).difference(numero))

#treino = pd.read_csv("bases/treino.csv")
#teste = pd.read_csv("bases/teste.csv")

for i in range(0,len(excluir)):
    nome = 'pob_0.combination' + str(excluir[i])
    nome1 = 'pob_1.combination' + str(excluir[i])
    nome2 = 'pob_2.combination' + str(excluir[i])   
    df_cop = df_cop.drop([nome,nome1,nome2],1)
    df_test_cop = df_test_cop.drop([nome,nome1,nome2],1)
      


prim = range(0,165,3)     
sec = range(1,165,3)     
terc = range(2,165,3)     

media = []
media1 = []
media2 = []
for i in range(0,len(df_cop)):
    media.append(np.mean(df_cop.iloc[i,prim]))
    media1.append(np.mean(df_cop.iloc[i,sec]))
    media2.append(np.mean(df_cop.iloc[i,terc]))
    
dat1_treino = pd.DataFrame([media,media1,media2])
dat1_treino = dat1_treino.T    
dat1_treino.columns = ['prob_10','prob_11','prob_22']

media = []
media1 = []
media2 = []



for i in range(0,len(df_test_cop)):
     print(i)
     temp_media = []
     temp_media1 = []
     temp_media2 = []
     for j in range(0,int(df_test_cop.shape[1] / 3)):
         if df_test_cop.iloc[i,prim[j]] != -999:
            temp_media.append(df_test_cop.iloc[i,prim[j]]) 
         if df_test_cop.iloc[i,sec[j]] != -999:
            temp_media1.append(df_test_cop.iloc[i,sec[j]])
         if df_test_cop.iloc[i,terc[j]] != -999:
            temp_media2.append(df_test_cop.iloc[i,terc[j]])
            
     media.append(np.mean(temp_media))
     media1.append(np.mean(temp_media1))
     media2.append(np.mean(temp_media2))
      
dat1_teste = pd.DataFrame([media,media1,media2])
dat1_teste = dat1_teste.T 
dat1_teste.columns = ['prob_10','prob_11','prob_22']


dat10 = np.asarray(dat1_treino)
df['prob_10'] = dat10[:,0]
df['prob_11'] = dat10[:,1]
df['prob_22'] = dat10[:,2]


dat10 = np.asarray(dat1_teste)
df_test['prob_10'] = dat10[:,0]
df_test['prob_11'] = dat10[:,1]
df_test['prob_22'] = dat10[:,2]






'''for i in range(0,len(df_test_cop)):
    media.append(np.mean(df_test_cop.iloc[i,prim]))
    media1.append(np.mean(df_test_cop.iloc[i,sec]))
    media2.append(np.mean(df_test_cop.iloc[i,terc]))


dat1_teste = pd.DataFrame([media,media1,media2])
dat1_teste = dat1_teste.T  '''



for k in range(0,len(dat)):
    print(k)
    
    agora.append(np.argmax(pd.value_counts(df_cop.iloc[k,numero])))

#df_cop['soma'] = agora
      
''' separacao '''

#df_test_cop['soma'] = -999
agora1 = []
contando = 0  
cont = []  
for k in range(0,len(df_test_cop)):
    #print(k)
    val = pd.value_counts(df_test_cop.iloc[k,numero])
    val_keys = val.keys()
    valmax = np.argmax(val)
    if valmax != -999:
       #print(k)
       contando = contando + 1
       agora1.append(valmax)
       cont.append(np.max(val))
    elif len(val_keys) > 1:
         #print(k)
         contando = contando + 1

         agora1.append(val_keys[1])
         cont.append(val[val_keys[1]])

    else:
          #print(k)
          contando = contando + 1

          agora1.append(valmax)

    
#df_test_cop['soma'] = agora1
 
        
            
            
'''from sklearn.ensemble import RandomForestClassifier

           


          
features_random  = ["bathrooms", "bedrooms", "price",'prob_0',
                    'month','week_month','agg_price_unique']
 

  
tt = df[features_random]
tt = tt.replace(np.inf, 0)
X_train1, X_test1, y_train1, y_test1 = train_test_split(tt, interest_level, test_size=0.20, random_state=7)
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train1, y_train1)
clf_probs = clf.predict_proba(X_test1)
score = log_loss(y_test1, clf_probs)'''








prob = managerprob(df)
   
agora10 = []
for i in enumerate(numero):
    gg = i[1]
    agora10.append(prob[gg])
    
valor_df  = df_cop.iloc[:,numero]
final = []
for j in range(0,len(df)):
    print(j)
    dic = []
    zero = 0
    um = 0
    dois = 0
    for k in range(0,len(nomes)):
        valp = agora10[k]
        val1 = valor_df.iloc[j][k]
        t1 = df.iloc[j][[nomes[k][0],nomes[k][1],nomes[k][2]]]
        valor = valp[(t1[0],t1[1],t1[2])]
        if val1 == 0:
           zero += valor
        if val1 == 1:
           um += valor
        if val1 == 2:
           dois += valor   
    dic.append(zero)
    dic.append(um)
    dic.append(dois)     
    final.append(np.argmax(dic))


teste_df = df_test_cop.iloc[:,numero]
final1 = []
for j in range(0,len(df_test)):
    print(j)
    dic = []
    zero = 0
    um = 0
    dois = 0
    for k in range(0,len(nomes)):
      val1 = teste_df.iloc[j][k]
      if val1 != -999:
        valp = agora10[k]
        #val1 = teste_df.iloc[j][k]
        t1 = df_test.iloc[j][[nomes[k][0],nomes[k][1],nomes[k][2]]]
        valor = valp[(t1[0],t1[1],t1[2])]
        if val1 == 0:
           zero += valor
        if val1 == 1:
           um += valor
        if val1 == 2:
           dois += valor 
    if zero != 0 or um != 0 or dois != 0:       
       dic.append(zero)
       dic.append(um)
       dic.append(dois)     
       final1.append(np.argmax(dic))
    else:
       final1.append(-999) 
    
    
'''import re
        

chaves = t1.keys()
valor =  df[(df['price'] > 3200) & (df['price'] <= 3250)]
pd.value_counts(valor['interest_level'])
gf = re.findall('\d+', chaves[0])'''

               

    



'''df['bins'] = pd.qcut(df['price'].values, 50)
df_test['bins'] = pd.qcut(df_test['price'].values, 50)'''



       
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions


xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 silent=1,
 min_child_weight=7,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

xgb_param = xgb1.get_xgb_params()
xgb_param['num_class'] = 3

xgtrain = xgb.DMatrix(X_train, label=y_train)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=10,
            metrics='mlogloss', early_stopping_rounds=150)
xgb1.set_params(n_estimators=cvresult.shape[0])
           
#alg.fit(X_train, y_train,eval_metric='mlogloss')
#dtrain_predprob = alg.predict_proba(X_train)
        



#clf1 = LogisticRegression(random_state=0)
clf2 = RandomForestClassifier(random_state=0)
clf3 = SVC(random_state=0, probability=True)
eclf = EnsembleVoteClassifier(clfs=[clf2, xgb1,clf3], weights=[1, 2,1], voting='soft')
       
gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))


for clf, lab, grd in zip([clf2, clf3, eclf],
                         ['Random Forest', 'RBF kernel SVM', 'Ensemble'],
                         itertools.product([0, 1], repeat=2)):
    clf.fit(X_train, y_train)
   

 

      
eclf.fit(X_train, y_train)

ty = df['num_photos'] - df['num_features']

df['ty'] = ty
valor1 = df[df['interest_level'] == 0]  
valor2 = df[df['interest_level'] == 1]  
valor3 = df[df['interest_level'] == 2]  
  
lista = ['building_id', 'display_address', 'manager_id','week_month']
  
  #lista1 = ['building_id', 'display_address', 'manager_id','bathrooms']
  import itertools
  tt = list(itertools.combinations(lista, 3))  


'''listau = []
for i in range(0,len(dic1)):
    listau.append(dic1[i][1])

clusterer = KMeans(n_clusters=50, init='k-means++', random_state=10) 
cluster_labels = clusterer.fit_predict(dic1)

unico = np.unique(cluster_labels)


for i in range(0,len(unico)):
  ind, = np.where(cluster_labels == unico[i])
  indes = []
  for j in range(0,len(ind)):
    indes.append(dic1[ind[j]])

  zero = 0
  um = 0
  dois = 0
    
  for k in range(0,len(indes)):
    
     dat = df[df['manager_id'] == indes[k][0]]
     val = pd.value_counts(dat['interest_level'])  
     val1 = val.keys()
     for p in range(0,len(val1)):
         if val1[p] == 0:
             zero = zero + val[0]
         if val1[p] == 1:
             um = um + val[1]
         if val1[p] == 2:
             dois = dois + val[2]


  print(zero)
  print(um)
  print(dois)
  print("\n")'''


  
listau = []

dic,dic10,dic20 = funcoes.porcentagem(df)
dic1 = sorted(dic.items(), key=lambda x:x[1])
dic10 = sorted(dic10.items(), key=lambda x:x[1])
dic20 = sorted(dic20.items(), key=lambda x:x[1])


dic50,dic51,dic52 = funcoes.porcentagem(df_test)
dic50 = sorted(dic50.items(), key=lambda x:x[1])
dic51 = sorted(dic51.items(), key=lambda x:x[1])
dic52 = sorted(dic52.items(), key=lambda x:x[1])


'''for i in range(0,len(dic1)):
    listau.append(dic1[i][1])'''

clusterer = KMeans(n_clusters=100, init='k-means++', random_state=10) 
cluster_labels = clusterer.fit_predict(dic1)

unico = np.unique(cluster_labels)

df['perce_manager_id'] = -999
for j in range(0,len(unico)):
  ind, = np.where(cluster_labels == unico[j])
  indes = []
  for i in range(0,len(ind)):
    indes.append(dic1[ind[i]])

  for i in range(0,len(indes)):
     df.loc[(df['manager_id'] == indes[i][0]),['perce_manager_id'] ] = unico[j]


df_test['perce_manager_id'] = -999
cluster_labels = clusterer.fit_predict(dic50)

for j in range(0,len(unico)):
  ind, = np.where(cluster_labels == unico[j])
  indes = []
  for i in range(0,len(ind)):
    indes.append(dic50[ind[i]])

  for i in range(0,len(indes)):
     df_test.loc[(df_test['manager_id'] == indes[i][0]),['perce_manager_id'] ] = unico[j]


# aqui comeca o per_address_id
     
clusterer = KMeans(n_clusters=100, init='k-means++', random_state=10) 
cluster_labels = clusterer.fit_predict(dic10)

unico = np.unique(cluster_labels)

df['perce_address_id'] = -999
  
for j in range(0,len(unico)):
  ind, = np.where(cluster_labels == unico[j])
  indes = []
  for i in range(0,len(ind)):
    indes.append(dic10[ind[i]])

  for i in range(0,len(indes)):
     df.loc[(df['display_address'] == indes[i][0]),['perce_address_id'] ] = unico[j]



'''clusterer = KMeans(n_clusters=25, init='k-means++', random_state=10) 
cluster_labels = clusterer.fit_predict(dic20)

unico = np.unique(cluster_labels)

df['perce_numphotos_id'] = -999
  
for j in range(0,len(unico)):
  ind, = np.where(cluster_labels == unico[j])
  indes = []
  for i in range(0,len(ind)):
    indes.append(dic20[ind[i]])

  for i in range(0,len(indes)):
     df.loc[(df['num_photos'] == indes[i][0]),['perce_numphotos_id']] = unico[j]'''

       
f1 = df[df['interest_level'] == 1]
f2 = df[df['interest_level'] == 2]
f3 = df[df['interest_level'] == 0]




#f1.groupby(['bathrooms', 'bedrooms']).bathrooms.count().groupby(level='bathrooms').count()
#f2.groupby(['bathrooms', 'bedrooms']).bathrooms.count().groupby(level='bathrooms').count()

'''train_df=pd.read_json('train.json')
test_df=pd.read_json('test.json')


index=list(range(train_df.shape[0]))
random.shuffle(index)
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(5):
    building_level={}
    for j in train_df['manager_id'].values:
        building_level[j]=[0,0,0]
    test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['manager_id']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
train_df['manager_level_low']=a
train_df['manager_level_medium']=b
train_df['manager_level_high']=c



a=[]
b=[]
c=[]
building_level={}
for j in train_df['manager_id'].values:
    building_level[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        building_level[temp['manager_id']][0]+=1
    if temp['interest_level']=='medium':
        building_level[temp['manager_id']][1]+=1
    if temp['interest_level']=='high':
        building_level[temp['manager_id']][2]+=1

for i in test_df['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0/sum(building_level[i]))
        b.append(building_level[i][1]*1.0/sum(building_level[i]))
        c.append(building_level[i][2]*1.0/sum(building_level[i]))
test_df['manager_level_low']=a
test_df['manager_level_medium']=b
test_df['manager_level_high']=c

features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')'''


total_features = 100
appen = []
for i in range(0,100):
    appen.append((math.sqrt(math.pow(100,2) - math.pow(i,2))) / 100)
    
    
    
    
df = pd.read_excel('quarterly.7775706.xls',sheetname='TB3MS')
# Creation of the variable spread
df['spread']=df['r5']-df['Tbill']
ax = df.plot(x='DATE',y='spread')   
ax.set_ylabel("Atmospheric pressure")






import pandas as pd
import seaborn as sns
sns.set(font="monospace")

# Load the brain networks example dataset
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# Select a subset of the networks
used_networks = [4, 6, 5, 9, 11, 12, 13, 16, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

# Create a custom palette to identify the networks
network_pal = sns.cubehelix_palette(len(used_networks),
                                    light=.9, dark=.1, reverse=True,
                                    start=1, rot=-2)
network_lut = dict(zip(map(str, used_networks), network_pal))

# Convert the palette to vectors that will be drawn on the side of the matrix
networks = df.columns.get_level_values("network")
network_colors = pd.Series(networks, index=df.columns).map(network_lut)

# Create a custom colormap for the heatmap values
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)

# Draw the full plot
par = sns.clustermap(df.corr(), row_colors=network_colors, linewidths=.5,
               col_colors=network_colors, figsize=(13, 13), cmap=cmap)  





import numpy as np
import seaborn as sns

sns.set()

# Create a random dataset across several variables
rs = np.random.RandomState(0)
n, p = 40, 8
d = rs.normal(0, 3, (n, p))
d += np.log(np.arange(1, p + 1)) * -5 + 11

# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(p, rot=-.5, dark=.3)

# Show each distribution with both violins and points
par = sns.violinplot(data=d, palette=pal, inner="points")







import numpy as np
import seaborn as sns

sns.set()

# Create a random dataset across several variables
rs = np.random.RandomState(0)
n, p = 40, 8
d = rs.normal(0, 3, (n, p))
d += np.log(np.arange(1, p + 1)) * -5 + 11

# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(p, rot=-.5, dark=.3)

# Show each distribution with both violins and points
par = sns.violinplot(data=d, palette=pal, inner="points")
par.set_ylabel('Temperature')


