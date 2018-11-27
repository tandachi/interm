# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 21:07:17 2016

@author: victor
"""
import numpy as np
import random
import decimal

import timeit
import random
import AvaliadordeSolucao
import Solucao
reload(AvaliadordeSolucao)
reload(Solucao)
from Solucao import Solucao
from AvaliadordeSolucao import AvaliadordeSolucao

class PSO_1(object):

    
    def __init__(self,populacao,w_ic, coef_1,coef_2,rand_1,rand_2,parametros, metrica_parada):
        self.id = 'None'
        self.metrica_parada = metrica_parada
        self.w = w_ic
        self.c1 = coef_1
        self.c2 = coef_2
        self.r1 = rand_1
        self.r2 = rand_2
        self.parametros = parametros
        self.velocidade = []
        self.populacao = populacao
        self.matrix_velocidade = np.zeros(shape=(populacao,parametros))
        self.matrix_best_individual = np.zeros(shape=(populacao,parametros+1))
        self.matrix_best_individual[:] = 'nan'
        self.best_global = np.zeros(shape=(1,parametros+1))
        self.best_global[:] = 'nan'
        self.mudou = 0
        self.qtd_mudou = 0
        self.inicial = 0
        self.inicial_1 = 0
    
    
    def criar_populacao(self,quantidade):
        pass
    
    
    def startSearch(self,p,n_ite):
        self.N = p.getNumAtributos()-1
        self.ava = AvaliadordeSolucao(p)
        self.t = timeit.default_timer()
        self.best = Solucao()
        
        self.evaluations = 0
        self.qtd_parada = 0
        
        while self.evaluations < n_ite and self.qtd_parada < self.metrica_parada:
          self.calculate_best_individual(valores)      
          self.calculate_best_global()
          if self.mudou == 1:
             self.qtd_parada = 0
          else:
             self.qtd_parada = self.qtd_parada + 1
            
          self.mudou = 0    
          self.update_velocity(valores)
       
          for j in range(0,len(valores)):
           
           valores[j].evaluate()
          
          self.evaluations = self.evaluations + 1
        
        ''' -----------------------------[implemente o método de busca a partir daqui]'''
        
        '''self.s = Solucao()
        self.s.solucao_1(self.N)
        self.s.initZero()
        self.best = Solucao()
        self.best.solucao_2(self.s)
        
        iter=0
        while iter < 1000:
            self.index = random.randint(0,self.N)
            self.s.inverte(self.index)
            self.ava.avalia(self.s)
            if self.s.getQuality() > self.best.getQuality():
                   self.best = Solucao()
                   self.best.solucao_2(self.s)
            else:
                self.s.inverte(self.index)
            print "Iter"+(iter) + ":\t"+self.best.getQuality()'''
        
        '''-------------------------[Fim do métoo de busca]'''
        
        '''t = timeit.default_timer() - t
        r = Result()
        r.setSolucao(best);
        r.setTime(t);
        r.setMetodo(this.getClass().getName());
        r.setDataset(p.getInstances().relationName());
        r.setCalls(ava.getCalls());
        r.setEvolucao(ava.getEvolucao());
        return r;   '''     
        
        
    ''' def iteracao(self,valores,n_ite):
      self.evaluations = 0
      self.qtd_parada = 0
      while self.evaluations < n_ite and self.qtd_parada < self.metrica_parada:
        self.calculate_best_individual(valores)      
        self.calculate_best_global()
        if self.mudou == 1:
            self.qtd_parada = 0
        else:
            self.qtd_parada = self.qtd_parada + 1
            
        self.mudou = 0    
        self.update_velocity(valores)
       
        for j in range(0,len(valores)):
           
           valores[j].evaluate()
          
        self.evaluations = self.evaluations + 1'''
   
    def update_velocity(self,valores):
       
      for i in range(0,self.populacao):
          for j in range(0,self.parametros):
              result = self.w * self.matrix_velocidade[i,j] + self.c1 * self.r1 * (self.matrix_best_individual[i,j] 
                                 - valores[i].valor_final[0,j]) + self.c2*self.r2*(self.best_global[0,j] - valores[i].valor_final[0,j] ) 
                                
              self.matrix_velocidade[i,j] = result              
              valores[i].valor_final[0,j] = valores[i].valor_final[0,j] + result
          
      self.r1 = round(random.uniform(0,1),2) 
      self.r2 = round(random.uniform(0,1),2)
              
    
  
    def calculate_best_individual(self,valores):
        
        if self.inicial == 0:
           for i in range(0,self.populacao):
               for j in range(0,self.parametros+1):
                   self.matrix_best_individual[i,j] = valores[i].valor_final[0,j]
           self.inicial = 1
           
        else:
            for i in range(0,self.populacao):
              if valores[i].valor_final[0,self.parametros] < self.matrix_best_individual[i,self.parametros]:
                 
                 for j in range(0,self.parametros+1):
                   self.matrix_best_individual[i,j] = valores[i].valor_final[0,j]
        
        
           
    
    def calculate_best_global(self):
        if self.inicial_1 == 0:
            for i in range(0,self.parametros+1):
                self.best_global[0,i] = self.matrix_best_individual[0,i]
            self.inicial_1 = 1
            
             
        for i in range(0,self.populacao):
            if self.matrix_best_individual[i,self.parametros] < self.best_global[0,self.parametros]:
                self.mudou = 1
                self.qtd_mudou = self.qtd_mudou + 1
                
                for j in range(0,self.parametros+1):
                    self.best_global[0,j] = self.matrix_best_individual[i,j] 
            
                
                
              
    def __repr__(self):
        return '<{}: {}>\n'.format(self.__class__.__name__, self.id)








