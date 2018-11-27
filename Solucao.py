# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 13:49:03 2016

@author: victor
"""
import numpy as np


class Solucao(object):
    
    def __init__(self):
        self.id = 'None'
        
     
    
    def solucao_1(self,n):
     
       self.data = np.zeros(shape=(1,n))
       self.quality = 0.0
    
    def solucao_2(self,t):
        self.data = t.data
        self.quality = t.getQuality()
        
    def get(self,n):
        return self.data[0,n]
     
    def sett(self,n,v):
        if v==0 or v==1:
            self.data[0,n] = v
        else:
             print "warning: use os valores 0 ou 1"
     
    def getData(self):
        return self.data
  
    
    def initRandom(self):
        pass
    
    def getQuality(self):
        return self.quality
    
    def setQuality(self,quality):
        self.quality = quality
        
    def setData(self,data):
        self.data = data
        
    def inverte(self,n):
        if self.data[0,n] == 1:
            self.data[0,n] = 0
        else:
            self.data[0,n] = 1
    
    def getBinaryFormar(self):
        pass
    
    def initOne(self):
        for i in range(0,len(self.data)):
            self.data[0,i] = 1
    
    def initZero(self):
         for i in range(0,len(self.data)):
             self.data[0,i] = 0
    

    def igual(self,candidate):
        for i in range(0,len(self.data)):
            if self.data[0,i] != candidate.data[0,i]:
                return False
        return True        
            
            
            
            
        
            
              
           
            




         