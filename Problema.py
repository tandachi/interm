# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 12:44:59 2016

@author: victor
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
'''from weka.core.converters import Loader
import weka.core.converters as converters
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.filters import Filter
from weka.classifiers import FilteredClassifier'''


class Problema(object):
    
    def __init__(self, path):
         self.path = path
         try:
           self.data = pd.read_csv(self.path,sep=',')
           #self.data = normalize(self.data)
           #self.data = pd.DataFrame(self.data)
           self.numAtributos = self.data.shape[1]
           self.numExemplos = self.data.shape[0]
           
           '''self.data = converters.load_any_file(self.path)
           self.data.class_is_last()
           self.numAtributos = self.data.num_attributes
           self.numExemplos = self.data.num_instances'''
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
        
    def getAttributeQuality(self):
        pass
    
    def w(self,i,j):
        return self.R[i,j]
        
    
    def somaW(self):
        pass
    
    def somaWK(self):
        pass
    
    
    
    
    
    
    
    