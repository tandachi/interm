# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 12:26:48 2016

@author: victor
"""
import timeit
import random
import AvaliadordeSolucao
import Solucao
reload(AvaliadordeSolucao)
reload(Solucao)
from Solucao import Solucao
from AvaliadordeSolucao import AvaliadordeSolucao
import numpy as np
from decimal import Decimal

class PSO_extend(object):
    
    
    def __init__(self,w,c1,c2,p,metrica_parada = 100):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.tam_pop = 30
        self.interacoes = 100
        self.r1 = round(random.uniform(0,1),2) 
        self.r2 = round(random.uniform(0,1),2)
        self.N = p.getNumAtributos()
        self.ava = AvaliadordeSolucao(p,self.tam_pop)
        self.velocidade = np.zeros(shape=(self.tam_pop,self.N))
        self.best_individual = np.zeros(shape=(self.tam_pop,self.N+1))
        self.best_individual[:] = 'nan'
        self.best_global = np.zeros(shape=(1,self.N+1))
        self.best_global[:] = 'nan'
        self.evaluations = 0
        self.qtd_parada = 0
        self.count_global = 0;
        self.contando = []
        self.qtd_mudou = 0
        self.mudou = 0
        self.inicial = 0
        self.inicial_1 = 0
        self.metrica_parada = metrica_parada
        self.solucoes = np.zeros(shape=(self.interacoes+1,self.N+1))
    
    def criar_populacao(self,tamanho,atributos):
        self.lower = (tamanho/3)*2
        populacao = np.zeros(shape=(tamanho,atributos+1))
        
        for i in range(0,self.lower):
           qtd_lower = random.sample(range(atributos),int(round(atributos*0.1)))
           
           for j in enumerate(qtd_lower):
               populacao[i,j[1]] = round(Decimal(random.uniform(0.01, 1.0)),4)
        
        for i in range(self.lower,tamanho):
            qtd_high = random.sample(range(atributos),random.randint((atributos/2 + 1),atributos))
            
            for j in enumerate(qtd_high):
                populacao[i,j[1]] = round(Decimal(random.uniform(0.01, 1.0)),4)
        
        return populacao
        
    def criar_populacao_pso_1(self,tamanho,atributos):
        self.lower = (tamanho/3)*2
        populacao = np.zeros(shape=(tamanho,atributos+1))
        
        for i in range(0,tamanho):
           qtd_lower = random.sample(range(atributos),random.randint(1,atributos))
           
           for j in enumerate(qtd_lower):
               populacao[i,j[1]] = round(Decimal(random.uniform(0.61, 1.0)),4)
        
        return populacao   
        
    

    def criar_populacao_randomica(self,tamanho,atributos):
        #self.lower = (tamanho/3)*2
        populacao = np.zeros(shape=(tamanho,atributos+1))
        
        for i in range(tamanho):
            qtd = random.sample(range(atributos),random.randint(0,atributos))
            
            for j in enumerate(qtd):
                populacao[i,j[1]] = round(Decimal(random.uniform(0.01, 1.0)),4)
        
        return populacao
    
    
    
    def startSearch(self,initialisation,funcao,clust):
        
        self.t = timeit.default_timer()
        self.best = Solucao()
        iter=0
        if initialisation == 4:
           self.populacao = self.criar_populacao(self.tam_pop,self.N)
        else:
            if initialisation == 1:
               self.populacao = self.criar_populacao_randomica(self.tam_pop,self.N)  
        
        
        ''' -----------------------------[implemente o método de busca a partir daqui]'''
        if funcao == 2:
           while iter <= self.interacoes and self.qtd_mudou < self.metrica_parada:
              print(iter) 
              
              
              self.populacao = self.ava.avalia(self.populacao,clust)
              
              self.calculate_best_individual(self.populacao)
              
              self.calculate_best_global()
              print("quantidade de features do melhor:")
              print(self.count_features(self.best_global[0,:]))
              print("score:")
              print(self.best_global[0,self.N])
              self.solucoes[iter,:] = self.best_global
              
              self.update_velocity()
              
              
              iter = iter + 1
              print("\n")
           for i in range(0,len(self.populacao)):
               self.contando.append(self.count_features(self.best_individual[i,:]))
               
        if funcao == 1:
            
           while iter < 100 and self.qtd_mudou < self.metrica_parada:
              print(iter)  
              self.populacao = self.ava.avalia(self.populacao)
              self.calculate_best_individual_pso_1_1(self.populacao)      
              self.calculate_best_global_pso_1_1()
              self.update_velocity()
              iter = iter + 1
          
           for i in range(0,len(self.populacao)):
               self.contando.append(self.count_features(self.best_individual[i,:]))
         
        if funcao == 3:
            
           while iter < 100 and self.qtd_mudou < self.metrica_parada:
              print(iter)  
             
              self.populacao = self.ava.avalia(self.populacao)
              
              self.calculate_best_individual_pso_4_3(self.populacao)
              
              self.calculate_best_global_pso_4_3()
              
              self.update_velocity()
              
              iter = iter + 1
          
           for i in range(0,len(self.populacao)):
               self.contando.append(self.count_features(self.best_individual[i,:])) 
         
        if funcao == 4:
            
          while iter <= self.interacoes and self.qtd_mudou < self.metrica_parada:
              print(iter) 
              
              
              self.populacao = self.ava.avalia(self.populacao,clust)
              
              self.calculate_best_individual_canonica(self.populacao)
              
              self.calculate_best_global_canonica()
              print("quantidade de features do melhor:")
              print(self.count_features(self.best_global[0,:]))
              print("score:")
              print(self.best_global[0,self.N])
              self.solucoes[iter,:] = self.best_global
              
              self.update_velocity()        
              iter = iter + 1
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
            #print("shape:",valores.shape)
            for i in range(0,len(valores)):
              #print("comparando particula:",i)
  
              count_particula = self.count_features(self.populacao[i,:])
              #print("qtd particula",count_particula)
              if count_particula > 0:
                #print("comparando particula1:",i)  
                #print(valores[i,self.N])
                #print(self.best_individual[i,self.N])
                if valores[i,self.N] > self.best_individual[i,self.N]:
                   #print("entrei_aqui_1")
                   for j in range(0,self.N+1):
                     self.best_individual[i,j] = valores[i,j]
                   continue 
             
              #count_particula = self.count_features(self.populacao[i,:])
                count_best_individual = self.count_features(self.best_individual[i,:])
                #if count_particula > 0:
                if valores[i,self.N] == self.best_individual[i,self.N] and count_particula < count_best_individual:
                    #print("entrei_aqui_2")

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
            
            count_best_individual = self.count_features(self.best_individual[i,:])
            if count_best_individual > 0:
              if self.best_individual[i,self.N] > (1.05 * self.best_global[0,self.N]):
                 #print("indice",i) 
                 self.mudou = 1
                 self.count_global = 0 
                 for j in range(0,self.N+1):
                      self.best_global[0,j] = self.best_individual[i,j]  
                 self.count_global = self.count_features(self.best_global[0,:])
                 continue
            
              #count_best_individual = self.count_features(self.best_individual[i,:])
            
              if self.best_global[0,self.N] == self.best_individual[i,self.N] and count_best_individual < self.count_global:
                  #print("indice",i)
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
        
  
    def calculate_best_individual_pso_1_1(self,valores):
        
        
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
        
        
    def calculate_best_global_pso_1_1(self):
        
      if self.inicial_1 == 0:
             for i in range(0,self.N+1):
                 self.best_global[0,i] = self.best_individual[0,i]
             self.inicial_1 = 1
            
             
      for i in range(0,len(self.populacao)):
             if self.best_individual[i,self.N] > self.best_global[0,self.N]:
                 self.mudou = 1
             for j in range(0,self.N+1):
                    self.best_global[0,j] = self.best_individual[i,j]        
         
      if self.mudou == 1:
         self.qtd_mudou = 0
      else:
         self.qtd_mudou = self.qtd_mudou + 1
            
      self.mudou = 0
     

    def calculate_best_individual_pso_4_3(self,valores):
        
        
        if self.inicial == 0:
           for i in range(0,len(valores)):
               for j in range(0,self.N+1):
                   self.best_individual[i,j] = valores[i,j]
           self.inicial = 1
           
        else:
            for i in range(0,len(valores)):
              count_particula = self.count_features(self.populacao[i,:])
              count_best_individual = self.count_features(self.best_individual[i,:])
              if count_particula > 0:
                if valores[i,self.N] > self.best_individual[i,self.N] and count_particula <= count_best_individual:
                 
                   for j in range(0,self.N+1):
                     self.best_individual[i,j] = valores[i,j]
                   break
             
              
              if count_particula > 0:
                if valores[i,self.N] == self.best_individual[i,self.N] and count_particula < count_best_individual:
                    
                    print("Entrei no best_individual")
                    for j in range(0,self.N+1):
                       self.best_individual[i,j] = valores[i,j]  
        
        
    def calculate_best_global_pso_4_3(self):
        if self.inicial_1 == 0:
            
            for i in range(0,self.N+1):
                self.best_global[0,i] = self.best_individual[0,i]
            self.inicial_1 = 1
            self.count_global = self.count_features(self.best_global[0,:])
             
        for i in range(0,len(self.populacao)):
            count_best_individual = self.count_features(self.best_individual[i,:])
            if self.best_individual[i,self.N] > self.best_global[0,self.N] and count_best_individual <= self.count_global:
               self.mudou = 1
               self.count_global = 0 
               for j in range(0,self.N+1):
                    self.best_global[0,j] = self.best_individual[i,j]  
               self.count_global = self.count_features(self.best_global[0,:])
               break
            
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
        
    
    def calculate_best_individual_pso_4_4(self,valores):
        
        
        if self.inicial == 0:
           for i in range(0,len(valores)):
               for j in range(0,self.N+1):
                   self.best_individual[i,j] = valores[i,j]
           self.inicial = 1
           
        else:
            for i in range(0,len(valores)):
              count_particula = self.count_features(self.populacao[i,:])
              count_best_individual = self.count_features(self.best_individual[i,:])
              if count_particula > 0:
                if valores[i,self.N] > self.best_individual[i,self.N] and count_particula <= count_best_individual:
                 
                   for j in range(0,self.N+1):
                     self.best_individual[i,j] = valores[i,j]
                   break
             
              
              if count_particula > 0:
                if valores[i,self.N] > 0.95 * self.best_individual[i,self.N] and count_particula < count_best_individual:
                    
                    print("Entrei no best_individual")
                    for j in range(0,self.N+1):
                       self.best_individual[i,j] = valores[i,j]
     
     

    def calculate_best_global_pso_4_4(self):
        if self.inicial_1 == 0:
            
            for i in range(0,self.N+1):
                self.best_global[0,i] = self.best_individual[0,i]
            self.inicial_1 = 1
            self.count_global = self.count_features(self.best_global[0,:])
             
        for i in range(0,len(self.populacao)):
            count_best_individual = self.count_features(self.best_individual[i,:])
            if self.best_individual[i,self.N] > self.best_global[0,self.N] and count_best_individual <= self.count_global:
               self.mudou = 1
               self.count_global = 0 
               for j in range(0,self.N+1):
                    self.best_global[0,j] = self.best_individual[i,j]  
               self.count_global = self.count_features(self.best_global[0,:])
               break
            
            if self.best_global[0,self.N] > 0.95 * self.best_individual[i,self.N] and count_best_individual < self.count_global:
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
              
              if particula[i] > 0.0:
                  
                  count = count + 1
         return count         
                  
         
    def calculate_best_individual_original(self,valores):
        
        
        if self.inicial == 0:
           for i in range(0,len(valores)):
               for j in range(0,self.N+1):
                   self.best_individual[i,j] = valores[i,j]
           self.inicial = 1
           
        else:
            #print("shape:",valores.shape)
            for i in range(0,len(valores)):
              #print("comparando particula:",i)
  
              count_particula = self.count_features(self.populacao[i,:])
              #print("qtd particula",count_particula)
              if count_particula > 0:
                #print("comparando particula1:",i)  
                #print(valores[i,self.N])
                #print(self.best_individual[i,self.N])
                if valores[i,self.N] > self.best_individual[i,self.N]:
                   #print("entrei_aqui_1")
                   for j in range(0,self.N+1):
                     self.best_individual[i,j] = valores[i,j]
                   continue 
             
              #count_particula = self.count_features(self.populacao[i,:])
                count_best_individual = self.count_features(self.best_individual[i,:])
                #if count_particula > 0:
                if valores[i,self.N] == self.best_individual[i,self.N] and count_particula < count_best_individual:
                    #print("entrei_aqui_2")

                    print("Entrei no best_individual")
                    for j in range(0,self.N+1):
                       self.best_individual[i,j] = valores[i,j]  
        
        
    def calculate_best_global_original(self):
        if self.inicial_1 == 0:
            
            for i in range(0,self.N+1):
                self.best_global[0,i] = self.best_individual[0,i]
            self.inicial_1 = 1
            self.count_global = self.count_features(self.best_global[0,:])
             
        for i in range(0,len(self.populacao)):
            
            count_best_individual = self.count_features(self.best_individual[i,:])
            if count_best_individual > 0:
              if self.best_individual[i,self.N] > self.best_global[0,self.N]:
                 #print("indice",i) 
                 self.mudou = 1
                 self.count_global = 0 
                 for j in range(0,self.N+1):
                      self.best_global[0,j] = self.best_individual[i,j]  
                 self.count_global = self.count_features(self.best_global[0,:])
                 continue
            
              #count_best_individual = self.count_features(self.best_individual[i,:])
            
              if self.best_global[0,self.N] == self.best_individual[i,self.N] and count_best_individual < self.count_global:
                  #print("indice",i)
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
        
        
        
        
        
        
    def calculate_best_individual_canonica(self,valores):
        
        
        if self.inicial == 0:
           for i in range(0,len(valores)):
               for j in range(0,self.N+1):
                   self.best_individual[i,j] = valores[i,j]
           self.inicial = 1
           
        else:
            #print("shape:",valores.shape)
            for i in range(0,len(valores)):
              #print("comparando particula:",i)
  
              count_particula = self.count_features(self.populacao[i,:])
              #print("qtd particula",count_particula)
              if count_particula > 0:
                #print("comparando particula1:",i)  
                #print(valores[i,self.N])
                #print(self.best_individual[i,self.N])
                if valores[i,self.N] > self.best_individual[i,self.N]:
                   #print("entrei_aqui_1")
                   for j in range(0,self.N+1):
                     self.best_individual[i,j] = valores[i,j]
                 
                 
                
        
        
    def calculate_best_global_canonica(self):
        if self.inicial_1 == 0:
            
            for i in range(0,self.N+1):
                self.best_global[0,i] = self.best_individual[0,i]
            self.inicial_1 = 1
            self.count_global = self.count_features(self.best_global[0,:])
             
        for i in range(0,len(self.populacao)):
            
            count_best_individual = self.count_features(self.best_individual[i,:])
            if count_best_individual > 0:
              if self.best_individual[i,self.N] > self.best_global[0,self.N]:
                 #print("indice",i) 
                 self.mudou = 1
                 self.count_global = 0 
                 for j in range(0,self.N+1):
                      self.best_global[0,j] = self.best_individual[i,j]  
                 self.count_global = self.count_features(self.best_global[0,:])
                 
            
            
              
        if self.mudou == 1:
            self.qtd_mudou = 0
        else:
            self.qtd_mudou = self.qtd_mudou + 1
            
        self.mudou = 0    
   

    

     