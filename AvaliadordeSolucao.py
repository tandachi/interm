# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 13:29:42 2016

@author: victor
"""
'''from sklearn.cross_validation import cross_val_score'''
'''from sklearn.cross_validation import StratifiedKFold'''
'''from sklearn.naive_bayes import GaussianNB'''
import numpy as np
from decimal import Decimal
#import sompy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import math

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


'''from sklearn.neighbors import KNeighborsClassifier'''
'''from sklearn.metrics import accuracy_score'''

'''from weka.core.converters import Loader'''
'''import weka.core.converters as converters'''
'''from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.filters import Filter'''
'''from weka.classifiers import FilteredClassifier'''

class AvaliadordeSolucao(object):
     
     
    def __init__(self,p,pop):
        
        self.p = p
        self.nfolds=10
        self.calls=0
        self.evolucao = []
        self.best = 0.0
        self.pop = pop
        self.atributos = self.p.getNumAtributos() - 1
        
        self.data = self.p.getInstances()
        
        
    def avalia(self,populacao,clust):
        self.populacao = populacao
        tam = self.populacao.shape[1]-1
                                 
       
        
        for i in range(0,self.pop):
            
            lista = self.excluir_coluna(self.populacao[i,:])
            
            #print("Qtd de features analisadas", tam-len(lista))
            if len(lista) != tam:
                
                '''self.dados = self.data.drop(self.data.columns[[lista]],1)
                self.dados = np.asarray(self.dados)'''
                
                '''vb = int(np.sqrt(int(5*np.sqrt(len(self.dados)))))
                mapsize = [vb,vb] # 5*np.sqrt(len(dados_con))
                self.som = sompy.SOMFactory.build(self.dados, mapsize, mask=None, mapshape='planar', lattice='rect', 
                             normalization='var', initialization='random', neighborhood='gaussian', training='batch', name='sompy')
                self.som.train(n_job=1,verbose=None)'''
                valor = self.run_clustering(self.data,lista,clust)
                #print(round(valor,2))
                self.populacao[i,tam] = round(valor,2)
                
        return self.populacao
    
    


    def run_clustering(self,dados,lista,n_cluster):
        data = dados.drop(dados.columns[[lista]],1).copy()
        data = np.asarray(data)
        dados = np.asarray(dados)
        clusterer = KMeans(n_clusters=n_cluster, random_state=10).fit(data)
        labels = clusterer.labels_
        unique = np.unique(labels)
        soma = 0
        soma1 = 0
        
        for i in enumerate(unique):
            
            dat_clus = dados[np.where(labels == i[1])]
            media = np.mean(dat_clus,axis=0)
            
            soma = (soma) + (np.power(euclidean_distances(dat_clus,media),2).sum())  
        
        '''print(soma)'''
        soma = soma / len(dados)
        media_all = np.mean(dados,axis=0)
        for i in enumerate(unique):
            
            dat_clus = dados[np.where(labels == i[1])]
            media = np.mean(dat_clus,axis=0)
            
            soma1 = (soma1) + ((len(dat_clus) / float(len(data))) * (np.power(euclidean_distances(media,media_all),2)))
        #print("Qtd features:")    
        #print(data.shape[1])    
        #print(soma1 / float(soma))
        #((dados.shape[1] - data.shape[1]) / float((dados.shape[1] - 1)))
        conec = soma1 / float(soma)
        val = ((math.pow(np.power(dados.shape[1],2) - np.power(data.shape[1],2), 1/2.0)) / float(dados.shape[0]))
        val = 1 / (1 + np.exp(-val)) 
        #n_clu = 1 - ((math.log10(n_cluster)) / float(math.log10(np.sqrt(dados.shape[0]))))
        #print("valores:")
        #print(conec)
        #print(val)
        #print(n_clu)
        return (conec) * val
        
        
    def makeIndex(self,s):
        pass
    
    
    def getCalls(self):
        return self.calls
        
    
    def getEvolucao(self):
        return self.evolucao

    
    def excluir_coluna(self,s):
           lista = []
           '''nome = p.columns.values'''
           for i in range(0,len(s)-1):
               
               if s[i] <= 0.0:
                  lista.append(i)
                  
           '''for i in lista:       
                 p = p.drop(p.columns[[lista]],1) '''
                
           
           return lista
     
    def excluir_coluna2(self,s):
        lista = []
        for i in range(0,len(s)-1):
               
               if s[i] <= 0.0:
                  lista.append(i+1)
        return lista
    
   

    def criar_variavel(self,s):
        tam = len(s)        
        lista2 =  ''
           
        lista2 = lista2 + str(s[0])
        for i in range(1,tam):
           lista2 = lista2 + "," + str(s[i])
        
        return lista2
        
           
'''from sklearn.model_selection import StratifiedKFold
predicao = []
skf = StratifiedKFold(n_splits=10)
knn = KNeighborsClassifier(n_neighbors=1)
for train, test in skf.split(data, label):
    data_train = data.ix[train,:]
    y_train = label[train]
    data_test = data.ix[test,:]
    y_test = label[test]
    knn.fit(data_train,y_train)
    teste = knn.predict(data_test)
    perc = accuracy_score(y_test,teste)
    predicao.append(perc)'''
    
    


'''from sklearn.cross_validation import train_test_split


X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.3)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
teste = knn.predict(X_test)
accuracy_score(y_test,teste)'''

'''import weka.core.jvm as jvm
jvm.start()
from weka.core.converters import Loader
import weka.core.converters as converters
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.filters import Filter
from weka.classifiers import FilteredClassifier
data = converters.load_any_file("entrada_1/sonar.csv")
data.class_is_last()
cls = Classifier(classname="weka.classifiers.lazy.IBk")
cls.options = ["-K", "3"]
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
lista = ["1","2","3"]
remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "0"])
remove.inputformat(data)
filtered = remove.filter(data)

lista2 =  ''
lista2 = lista2 + "," + str(lista[0])'''


                


'''data = dados.drop(dados.columns[[lista]],1).copy()
        data = np.asarray(data)
        dados = np.asarray(dados)
        clusterer = KMeans(n_clusters=n_cluster, random_state=10).fit(data)
        labels = clusterer.labels_
        unique = np.unique(labels)
        soma = 0
        soma1 = 0
        
        for i in enumerate(unique):
            
            dat_clus = dados[np.where(labels == i[1])]
            media = np.mean(dat_clus,axis=0)
            
            soma = (soma) + (np.power(euclidean_distances(dat_clus,media),2).sum())  
        
        #print(soma)
        soma = soma / len(dados)
        media_all = np.mean(dados,axis=0)
        for i in enumerate(unique):
            
            dat_clus = dados[np.where(labels == i[1])]
            media = np.mean(dat_clus,axis=0)
            
            soma1 = (soma1) + ((len(dat_clus) / float(len(data))) * (np.power(euclidean_distances(media,media_all),2)))
            
            
        
        return((soma1 / float(soma)) * ((dados.shape[1] - data.shape[1]) / float((dados.shape[1] - 1))) *
            (1 - ((math.log10(n_cluster)) / float(math.log10(np.sqrt(dados.shape[0]))))))'''


'''app = []
app2 = []
app3 = []
for i in range(1,281):
  vald = (math.pow(np.power(281,2) - np.power(i,2),1/2.0)) / float(452)
  app3.append(1 / (1 + np.exp(-vald)))  
  app.append(((math.pow(np.power(281,2) - np.power(i,2),1/2.0)) / float(452)))
  app2.append(((math.pow(np.power(281,2) - np.power(i,2),1/3.0)) / float(452)))     
  #print(((np.sqrt(np.power(281,2) - np.power(i,2))) / float(452)))
  
app1 = []

for i in range(1,281):
    
  app1.append((281 - i) / float(280))   
  #print(((np.sqrt(np.power(281,2) - np.power(i,2))) / float(452)))
    
  
app = []
app2 = []
for i in range(1,281):
    
  vald = (math.pow(np.power(281,2) - np.power(i,2),1/2.0)) / float(452)
  app.append(1 / (1 + np.exp(-vald)))
  #app2.append(((math.pow(np.power(281,2) - np.power(i,2),1/4.0)) / float(452))) '''
  
  
  
  
  
  
  
  
          
                