# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 12:26:48 2016

@author: victor
"""
import timeit
import random
import AvaliadordeSolucao
reload(AvaliadordeSolucao)
from AvaliadordeSolucao import AvaliadordeSolucao
import numpy as np
from decimal import Decimal

class PSO_extend(object):
    
    
    def __init__(self,w,c1,c2,tam_pop, itera, metrica_parada = 100):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.tam_pop = 30
        self.inter = 100
        self.r1 = round(random.uniform(0,1),2) 
        self.r2 = round(random.uniform(0,1),2)
        self.evaluation = Evaluation()
        #self.population = np.zeros()
        #self.N = p.getNumAtributos()
        #self.ava = AvaliadordeSolucao(p,self.tam_pop)
        #self.velocidade = np.zeros(shape=(self.tam_pop,self.N))
        #self.best_individual = np.zeros(shape=(self.tam_pop,self.N+1))
        #self.best_individual[:] = 'nan'
        #self.best_global = np.zeros(shape=(1,self.N+1))
        #self.best_global[:] = 'nan'
        #self.evaluations = 0
        #self.qtd_parada = 0
        #self.count_global = 0;
        #self.contando = []
        #self.qtd_mudou = 0
        #self.mudou = 0
        #self.inicial = 0
        #self.inicial_1 = 0
        #self.metrica_parada = metrica_parada
        #self.solucoes = np.zeros(shape=(self.interacoes+1,self.N+1))
    
    def create_population(self,size,attri):
        
        lower = int((size/3)*2)
        pop = np.zeros(shape=(size,attri+4))
        vel = np.zeros(shape=(size,attri))
        
        for i in range(0,lower):
           qtd_lower = random.sample(range(attri),int(round(attri*0.2)))
           pop[i,qtd_lower] = 1
           pop[i,attri] = random.uniform(1,2)
           pop[i,attri+1] = random.uniform(0,1)
           
        for i in range(lower, size):
            qtd_high = random.sample(range(attri),random.randint((attri/2 + 1),attri))
            pop[i,qtd_high] = 1
            pop[i,attri] = random.uniform(1,2)
            pop[i,attri+1] = random.uniform(0,1)
            
            
        return pop, vel
        
   
        
           
    
    def fit(self, data, target, estimator, **params):
        
        fmin = 0
        fmax = 1
        theta = 0.5
        alpha = 0.5
        epsi = 0.5
        n_features = data.shape[1]
        pos_a = n_features
        pos_r = n_features + 1
        pop , vel = self.create_population(self.tam_pop, n_features)
        maxfit, maxindex = 0
        bestglobalfit = -1
        bestglobal = np.zeros(shape=(1,n_features))
        freq = np.random.uniform(low=0, high=1, size=self.tam_pop)
        iter = 0
        while iter <= self.inter and qtd_change < self.metric_change:
            pop = self.evaluation.fit(estimator, data, target, params)
            rand = np.random.uniform(low=0, high=1, size=self.tam_pop)
            for i in self.tam_pop:
                if (rand[i]  < pop[i, pos_a]) & (pop[i, -1] > pop[i, -2]):
                   pop[i, -2] =  pop[i, -1]
                   pop[i, pos_a] = theta * pop[i, pos_a]
                   pop[i, pos_r] = pop[i, pos_r] * (1 - np.exp(-alpha*iter))
                   
            maxindex = np.argmax(pop[:, -2])      
            maxfit = pop[maxindex, -2]
            if maxfit > bestglobalfit:
               bestglobalfit = maxfit
               for i in self.n_features:
                  bestglobal[:, i] = pop[maxindex, i] 
            
            beta = np.random.uniform(low=0, high=1, size=self.tam_pop)
            rand = np.random.uniform(low=0, high=1, size=self.tam_pop)
            rand_2 = np.random.uniform(low=0, high=1, size=self.tam_pop)
            ave_pos_a = np.mean(pop[:, pos_a])
            for i in self.tam_pop:
                if (rand[i]  < pop[i, pos_r]):
                   for j in self.n_features:
                       pop[i, j] = pop[i, j] + (epsi * ave_pos_a )
                       sigma = np.random.uniform(low=0, high=1, size=1)
                       if sigma < (1 * (1 / (1 + np.exp(-pop[i, j])))):
                           pop[i, j] = 1
                    
                       else:
                           pop[i, j] = 0
                       
                if (rand_2[i] < pop[i, pos_a]) & (pop[i, -2] < bestglobalfit):
                   freq = fmin + (fmax - fmin) * beta
                   for j in self.n_features:
                                     
                      vel[i, j] = vel[i, j] + (np.mean(vel[:,j]) - vel[i, j])* freq[i]
                      pop[i, j] = pop[i, j] + vel[i, j]
                      sigma = np.random.uniform(low=0, high=1, size=1)
                      if sigma < (1 * (1 / (1 + np.exp(-pop[i, j])))):
                           pop[i, j] = 1
                      else:
                           pop[i, j] = 0
       
    def update_velocity(self):
       
      for i in range(0,len(self.populacao)-1):
          for j in range(0,self.N):
              result = self.w * self.velocidade[i,j] + self.c1 * self.r1 * (self.best_individual[i,j] 
                                 - self.populacao[i,j]) + self.c2*self.r2*(self.best_global[0,j] - self.populacao[i,j]) 
                                
              self.velocidade[i,j] = result            
              '''self.populacao[i,j] = 1/(1+np.exp(-(self.populacao[i,j] + result)))'''
              self.populacao[i,j] = self.populacao[i,j] + result
              self.r1 = round(random.uniform(0,1),2) 
              self.r2 = round(random.uniform(0,1),2)
            
      
              
    
  
    def calculate_best_individual(self,valores):
        
        
        if self.inicial == 0:
           for i in range(0,len(valores)):
               for j in range(0,self.N+1):
                   self.best_individual[i,j] = valores[i,j]
           self.inicial = 1
           
        else:
            for i in range(0,len(valores)):
              if valores[i,self.N] > self.best_individual[i,self.N]:
                 
                 for j in range(0,self.N+1):
                   self.best_individual[i,j] = valores[i,j]
                 continue
             
              count_particula = self.count_features(self.populacao[i,:])
              count_best_individual = self.count_features(self.best_individual[i,:])
              if count_particula > 0:
                if valores[i,self.N] == self.best_individual[i,self.N] and count_particula < count_best_individual:
                    
                    print("Entrei no best_individual")
                    for j in range(0,self.N+1):
                       self.best_individual[i,j] = valores[i,j]  
        
        
    def calculate_best_global(self):
        if self.inicial_1 == 0:
            
            for i in range(0,self.N+1):
                self.best_global[0,i] = self.best_individual[0,i]
            self.inicial_1 = 1
            self.count_global = self.count_features(self.best_global[0,:])
             
        for i in range(0,len(self.populacao)):
            #print(i)
            if self.best_individual[i,self.N] > self.best_global[0,self.N]:
               self.mudou = 1
               self.count_global = 0 
               for j in range(0,self.N+1):
                    self.best_global[0,j] = self.best_individual[i,j]  
               self.count_global = self.count_features(self.best_global[0,:])
               
               continue
            
            count_best_individual = self.count_features(self.best_individual[i,:])
            
            if self.best_global[0,self.N] == self.best_individual[i,self.N] and count_best_individual < self.count_global:
                self.mudou = 1
                print("Entrei no best_global")
                self.count_global = 0
                for j in range(0,self.N+1):
                    self.best_global[0,j] = self.best_individual[i,j]  
                self.count_global = self.count_features(self.best_global[0,:])
         
        if self.mudou == 1:
            self.qtd_mudou = 0
        else:
            self.qtd_mudou = self.qtd_mudou + 1
            
        self.mudou = 0
        
  
    
        
    def count_features(self,particula):
         count = 0;
         for i in range(0,self.N):
              
              if particula[i] > 0.6:
                  count = count + 1
         return count         
                  
     
 




       
         
        