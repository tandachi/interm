# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 11:57:55 2016

@author: victor
"""
import Problema

import MeuMetodo
reload(Problema)
reload(MeuMetodo)
from Problema import Problema
from MeuMetodo import PSO_extend
import numpy as np
import os
import pandas as pd


         
def iniciar(base_treino,label_treino):         
         
    '''arquivos = os.listdir('entrada2/')

    for i in range(0,len(arquivos)):'''

    p = Problema(base_treino,label_treino)      
    metodo = PSO_extend(0.7298,1.49618,1.49618,p,metrica_parada = 50)
    metodo.startSearch(4,2)

    resultado = np.zeros(shape=(1,p.getNumAtributos() + 1))
    cont = np.zeros(shape=(1,1))

    resultado[0,:] = metodo.best_global
    cont[0,:] = metodo.count_features(metodo.best_global[0,:])


    '''r.printResult()'''
     
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
  
  

  
  
    '''np.savetxt("saida/resultado.txt", resultado, newline="\n")
    np.savetxt("saida/cont-.txt", cont, newline="\n")
    np.savetxt("saida/cont-total-.txt", metodo.solucoes, newline="\n")'''
   
  
  
    '''p1 = np.loadtxt("saida/features-cont-total-.txt")
    for i in range(0,len(p1)):
        print(pd.value_counts(p1[i,0:18] > 0.6))'''
      
      
      
    return metodo
  





  