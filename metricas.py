# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 14:45:30 2017

@author: rebli
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import math
import itertools


class metricas(object):
    
    def __init__(self):
        pass
        #self.bases = bases
        #self.labels = labels
        
    def clustering(self,data,n_cluster):
        
        clusterer = KMeans(n_clusters=n_cluster, random_state=10).fit(data)
        labels = clusterer.labels_
        
        return labels,clusterer.cluster_centers_
    
    def intra(self,data,pred_label,center_cluster):
        unicos = np.unique(pred_label)
        #data['label'] = pred_label
        intra = 0
        for i in range(0,len(unicos)):
            #print(i)
            ind, = np.where(pred_label == unicos[i])
            dat = data.iloc[ind,:]
            intra = intra + np.sum(euclidean_distances(dat,center_cluster[i]))
            
        return intra
    

    def class_purity(self,true_labels,pred_labels):
        unicos = np.unique(true_labels)
        fracoes = []
        for i in range(0,len(unicos)):
            #print(i)
            ind, = np.where(pred_labels == unicos[i])
            #ind = set(ind)
            clus = true_labels[ind]
            majority = pd.value_counts(clus).keys()[0]
            clus1, = np.where(true_labels == unicos[majority])
            clus1 = set(clus1)
            ind = set(ind)
            correto = len(ind & clus1)
            fracao = correto / float(len(clus1))
            fracoes.append(fracao)
        return fracoes    

    
    def FMeasure(self,true_labels,pred_labels):
        #unicos = np.unique(pred_labels)
        ran = range(0,len(pred_labels))
        tt = list(itertools.combinations(ran, 2))
        TP = 0
        FN = 0
        TN = 0
        FP = 0
        for i in range(0,len(tt)):
            clus1 = pred_labels[tt[i][0]]
            clus2 = pred_labels[tt[i][1]]
            clas1 = true_labels[tt[i][0]]
            clas2 = true_labels[tt[i][1]]
            if clas1 == clas2:
                if clus1 == clus2:
                    TP =  TP + 1
                else:
                    FN = FN + 1
            elif clas1 != clas2:
                if clus1 == clus2:
                    FP = FP + 1
                else:
                    TN = TN + 1
        
        precision = TP / float(TP + FP)
        recal =  TP / float(TP + FN)
        FMeasure = 2 * (precision * recal) / float((precision + recal))             
        return FMeasure          
        
        
    def fit(self,data,true_label,n_cluster):
        pred_label,center_clusters = self.clustering(data,n_cluster)
        #pred_label = pred_label + 1
        intra_value = self.intra(data,pred_label,center_clusters)
        
        purity = self.class_purity(true_label,pred_label)
        FM = self.FMeasure(true_label,pred_label)
        return intra_value,purity,FM
        
        
    def excluir_coluna(self,s):
           lista = []
           '''nome = p.columns.values'''
           for i in range(0,len(s)-1):
               
               if s[i] <= 0.0:
                  lista.append(i)
                  
           '''for i in lista:       
                 p = p.drop(p.columns[[lista]],1) '''
                
           
           return lista   
    
    
    
    
    
    
#data = dados.drop(dados.columns[[lista]],1).copy()



'''po = pd.read_csv("wine.csv")
po = po['0']
po.to_csv("wine.txt", index=False)'''

'''import metricas
reload(metricas)
from metricas import metricas
met = metricas()
intra, purity, fm = met.fit(data, label, 16) '''   