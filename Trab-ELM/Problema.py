# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 12:44:59 2016

@author: victor
"""

import pandas as pd
import numpy as np


class Problema(object):
    
    def __init__(self, path,label):
         
         try:
           self.data = path
           self.label = label
           
           #self.indices_treino = indices_treino
           #self.indices_validacao = indices_validacao
           #self.xgb = agl
           self.numAtributos = self.data.shape[1]
           self.numExemplos = self.data.shape[0]
           
           
         except Exception:
              print "Não foi Possivel abrir o arquivo" + self.path 
              
    def getNumAtributos(self):
        return self.numAtributos   

    def getNumExemplos(self):
        return self.numExemplos     
        
    def getInstances(self):
        return self.data
        
    def setInstances(self,data):
        self.data = data
        
  
    
    
    
    
    
    
    
    