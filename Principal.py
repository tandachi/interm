# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 11:57:55 2016

@author: victor
"""
import Problema
import PSO_1
import MeuMetodo
import SFS_SBS

reload(Problema)
reload(PSO_1)
reload(MeuMetodo)
reload(SFS_SBS)

from Problema import Problema
from PSO_1 import PSO_1
from MeuMetodo import PSO_extend
from SFS_SBS import SFS_SBS
import numpy as np
import os
#import pandas as pd
saidas = ['saida1/','saida2/','saida3/','saida4/','saida5/','saida6/','saida7/','saida8/','saida9/',
          'saida10/']
arquivos = os.listdir('entrada2/')
numb = 0
for j in range(0,10):
    
 for i in range(0,len(arquivos)): 
  #classes = [13,2,2,3,2,10,2,99,3]
  classes = [3]
  p = Problema("entrada2/"+arquivos[i])
  #bas = p.getInstances()
  #print(bas.shape)    
  metodo = PSO_extend(0.7298,1.49618,1.49618,p,metrica_parada = 90)
  metodo.startSearch(1,4,classes[i])

  resultado = np.zeros(shape=(1,p.getNumAtributos() + 1))
  cont = np.zeros(shape=(1,1))

  resultado[0,:] = metodo.best_global
  cont[0,:] = metodo.count_features(metodo.best_global[0,:])


  np.savetxt('results/conts/canonico/'+arquivos[i].split('.')[0]+'/'+arquivos[i].split('.')[0]+str(numb)+".txt", resultado, newline="\n")
  np.savetxt('results/conts_histo/canonico/'+arquivos[i].split('.')[0]+'/'+arquivos[i].split('.')[0]+str(numb)+"-cont-.txt", cont, newline="\n")
  np.savetxt('results/conts_total/canonico/'+arquivos[i].split('.')[0]+'/'+arquivos[i].split('.')[0]+str(numb)+"-cont-total-.txt", metodo.solucoes, newline="\n")

  '''p = Problema("entrada2/"+arquivos[i])          
  metodo = PSO_extend(0.7298,1.49618,1.49618,p,100)
  metodo.startSearch(1,1)
  resultado[1,:] = metodo.best_global
  cont[1,:] = metodo.count_features(metodo.best_global[0,:])'''

  '''p = Problema("entrada2/"+arquivos[i])              
  metodo = PSO_extend(0.7298,1.49618,1.49618,p,100)
  metodo.startSearch(4,3)
  resultado = np.zeros(shape=(4,p.getNumAtributos()))
  cont = np.zeros(shape=(4,1))

  resultado[0,:] = metodo.best_global
  cont[0,:] = metodo.count_features(metodo.best_global[0,:])'''

  '''p = Problema("entrada2/"+arquivos[i])           
  metodo = PSO_extend(0.7298,1.49618,1.49618,p,100)
  metodo.startSearch(4,4)'''
  
  

  
  
  '''np.savetxt(saidas[j]+arquivos[i].split('.')[0]+".txt", resultado, newline="\n")
  np.savetxt(saidas[j]+arquivos[i].split('.')[0]+"-cont-.txt", cont, newline="\n")
  np.savetxt(saidas[j]+arquivos[i].split('.')[0]+"-cont-total-.txt", metodo.solucoes, newline="\n")'''
  
  
  '''print("numero de features:", cont)
  print("indices das features:", np.where(resultado[0,0:15] > 0.6))'''
  
 numb = numb + 1      
  

''''sai = range(0,61)
ap = []
for i in enumerate(sai):
    #print(i[1])
    ap.append(np.sqrt((np.power(60,2) - np.power(i[1],2)) /59) / 208)   

plt.plot(sai,ap)
 
     
sai = range(0,61)
ap = []
for i in enumerate(sai):
    #print(i[1])
    ap.append(np.sqrt((np.power(60,2) - np.power(i[1],2))) / 208)   

plt.plot(sai,ap)
       
sai = range(0,61)
ap = []
for i in enumerate(sai):
    #print(i[1])
    ap.append((60 - i[1]) / 59)    
      



plt.plot(sai,ap)'''



