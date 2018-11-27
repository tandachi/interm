# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 10:43:58 2017

@author: rebli
"""


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import math
import itertools
import os
from sklearn import preprocessing
from tabulate import tabulate
import metricas
reload(metricas)
from metricas import metricas
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


diretorios = os.listdir('results/conts/canonico')

labels = os.listdir('labels/')
arquivos2 = os.listdir('entrada2/')
met = metricas()

# 2 - canonico, 3 - original, 4 - proposto

intra2 = []
purity2 = []
fm2 = []

classes = [13,2,2,2,10,2,99,3]
for p in range(0,len(diretorios)):
  print("Analisando diretorio:", diretorios[p])  
  arquivos = os.listdir('results/conts/canonico/'+diretorios[p]+"/")
  base = pd.read_csv("entrada2/"+arquivos2[p])
  label = np.loadtxt('labels/'+labels[p])
  le = preprocessing.LabelEncoder()
  le.fit(np.unique(label))
  label = le.transform(label)
  intra1 = []
  purity1 = []
  fm1 = []
  
  for i in range(0,len(arquivos)):
       print(i)
       po = np.loadtxt('results/conts/canonico/'+ diretorios[p] + '/' + arquivos[i])
       liv = met.excluir_coluna(po)
       base1 = base.drop(base.columns[[liv]],1).copy()
       intra, purity, fm = met.fit(base1, label, classes[p])
       intra1.append(intra)
       purity1.append(purity)
       fm1.append(fm)
  intra2.append(intra1)
  purity2.append(purity1)
  fm2.append(fm1)
 
#aqui    

somas_proposto = []
diretorios_conts = os.listdir('bases_unicos_conts/proposto/')
for p in range(0,len(diretorios_conts)):
  print("Analisando diretorio:", diretorios_conts[p])  
  arquivos = os.listdir('bases_unicos_conts/proposto/'+diretorios_conts[p]+"/")
  #base = pd.read_csv("entrada2/"+arquivos2[p])
  #label = np.loadtxt('labels/'+labels[p])
  #le = preprocessing.LabelEncoder()
  #le.fit(np.unique(label))
  #label = le.transform(label)
  #intra1 = []
  #purity1 = []
  #fm1 = []
  soma = 0
  for i in range(0,len(arquivos)):
       #print(i)
       po = np.loadtxt('bases_unicos_conts/proposto/'+ diretorios_conts[p] + '/' + arquivos[i])
       soma = soma + po
       #liv = met.excluir_coluna(po)
       #base1 = base.drop(base.columns[[liv]],1).copy()
       #intra, purity, fm = met.fit(base1, label, classes[p])
       #intra1.append(intra)
       #purity1.append(purity)
       #fm1.append(fm)
  soma = int(soma / 10)     
  somas_proposto.append(soma)
  



somas_proposto1 = []
diretorios_conts = os.listdir('bases_unicos/proposto/')
for p in range(0,len(diretorios_conts)):
  print("Analisando diretorio:", diretorios_conts[p])  
  arquivos = os.listdir('bases_unicos/proposto/'+diretorios_conts[p]+"/")
  #base = pd.read_csv("entrada2/"+arquivos2[p])
  #label = np.loadtxt('labels/'+labels[p])
  #le = preprocessing.LabelEncoder()
  #le.fit(np.unique(label))
  #label = le.transform(label)
  #intra1 = []
  #purity1 = []
  #fm1 = []
  soma = 0
  for i in range(0,len(arquivos)):
       #print(i)
       po = np.loadtxt('bases_unicos/proposto/'+ diretorios_conts[p] + '/' + arquivos[i])
       soma = soma + po[len(po)-1]
       #liv = met.excluir_coluna(po)
       #base1 = base.drop(base.columns[[liv]],1).copy()
       #intra, purity, fm = met.fit(base1, label, classes[p])
       #intra1.append(intra)
       #purity1.append(purity)
       #fm1.append(fm)
  soma = soma / 10     
  somas_proposto1.append(soma)




#canonico
arry_intra2 = []
for i in range(0,len(intra2)):
    arry_intra2.append(np.mean(intra2[i]))

arry_fm2 = []
for i in range(0,len(fm2)):
    arry_fm2.append(np.mean(fm2[i]))

 
arry_pur2 = []
for i in range(0,len(purity2)):
    arry_temp = []    
    for j in range(0,len(purity2[i])):
        arry_temp.append(np.mean(purity2[i][j]))     
   
    arry_pur2.append(np.mean(arry_temp))

arr2 = pd.DataFrame([somas_canonico1,somas_canonico,arry_intra2,arry_fm2,arry_pur2])
f = open('canonico.txt', 'w')
f.write(tabulate(arr2, headers=['arrhy', 'h-va-wi-no', 'h-va-without-no'
                                   ,'lu-can', 'Musk','optdi', 'sonar','specs-leaf','wine']))
f.close()    
    

    



#original
arry_intra3 = []
for i in range(0,len(intra3)):
    arry_intra3.append(np.mean(intra3[i]))

arry_fm3 = []
for i in range(0,len(fm3)):
    arry_fm3.append(np.mean(fm3[i]))

 
arry_pur3 = []
for i in range(0,len(purity3)):
    arry_temp = []    
    for j in range(0,len(purity3[i])):
        arry_temp.append(np.mean(purity3[i][j]))     
   
    arry_pur3.append(np.mean(arry_temp))    
    
 
arr3 = pd.DataFrame([somas_original1,somas_original,arry_intra3,arry_fm3,arry_pur3])
f = open('original.txt', 'w')
f.write(tabulate(arr3, headers=['arrhy', 'h-va-wi-no', 'h-va-without-no'
                                   ,'lu-can', 'Musk','optdi', 'sonar','specs-leaf','wine']))
f.close()       
    

#proposto
arry_intra4 = []
for i in range(0,len(intra4)):
    arry_intra4.append(np.mean(intra4[i]))

arry_fm4 = []
for i in range(0,len(fm4)):
    arry_fm4.append(np.mean(fm4[i]))

 
arry_pur4 = []
for i in range(0,len(purity4)):
    arry_temp = []    
    for j in range(0,len(purity4[i])):
        arry_temp.append(np.mean(purity4[i][j]))     
   
    arry_pur4.append(np.mean(arry_temp))        
    


arr4 = pd.DataFrame([somas_proposto1,somas_proposto,arry_intra4,arry_fm4,arry_pur4])
f = open('proposto.txt', 'w')
f.write(tabulate(arr4, headers=['arrhy', 'h-va-wi-no', 'h-va-without-no'
                                   ,'lu-can', 'Musk','optdi', 'sonar','specs-leaf','wine']))
f.close()      



