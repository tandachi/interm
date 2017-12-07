# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 19:01:18 2017

@author: rebli
"""

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn import preprocessing
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import matplotlib.pylab as plt
# import sompy as sompy
import pandas as pd
import numpy as np
from time import time
import sompy


def calcula_pca_visualizacao(dados,fim_normal,inicio_falha,falha):
    
  dados.ix[0:fim_normal:,18] = 0    
  dados.ix[inicio_falha:1200,18] = falha  
  valor_negativo = len(dados[dados.ix[:,18] == falha]) - 1
  file_label = dados.ix[:,18]
  dados = dados.drop(18,1)
  dim = dados.shape
  file_amostras_normal = dados[0:fim_normal:]

  mean_normal = np.mean(dados[0:fim_normal:])
  sd_normal = np.std(dados[0:fim_normal:])
  file_amostras_falha = dados[inicio_falha::]

  dim_1 = file_amostras_falha.shape


  file_amostras_normal = file_amostras_normal - mean_normal
  file_amostras_normal = file_amostras_normal - sd_normal


  file_amostras_falha = file_amostras_falha - mean_normal
  file_amostras_falha = file_amostras_falha - sd_normal

  merge_dataframe = [file_amostras_normal,file_amostras_falha]
  dataframe_merge = pd.concat(merge_dataframe)
  
  cv = np.cov(file_amostras_normal.T)
  w,v = np.linalg.eig(cv)
  eig_pairs = [(np.abs(w[i]),v[:,i]) for i in range(len(w))]
  eig_pairs.sort(key=lambda x: x[0], reverse=True)
  matrix_w = np.hstack((eig_pairs[0][1].reshape(18,1), eig_pairs[1][1].reshape(18,1),eig_pairs[2][1].reshape(18,1)))
  transformed = dataframe_merge.dot(matrix_w)
  transformed = transformed.as_matrix()
  transformed = np.float32(transformed)
  pos = file_label[file_label == 0].index.tolist()
  neg = file_label[file_label == falha].index.tolist()
  neg = neg[0:valor_negativo]

  '''fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, projection='3d')
  plt.rcParams['legend.fontsize'] = 10   
  ax.scatter(transformed[pos,0], transformed[pos,1], transformed[pos,2],c='r',marker='o')
  ax.scatter(transformed[neg,0], transformed[neg,1], transformed[neg,2],c='b',marker='+')
  
  texto = 'Samples for falha normal and falha',str(falha),'falha comeca em',fim_normal
      
  plt.title(texto)
  ax.legend(loc='upper right')
  
  plt.show()'''
  return transformed
  
  

dlen = 200
Data1 = pd.DataFrame(data= 1*np.random.rand(dlen,2))
Data1.values[:,1] = (Data1.values[:,0][:,np.newaxis] + .42*np.random.rand(dlen,1))[:,0]


Data2 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+1)
Data2.values[:,1] = (-1*Data2.values[:,0][:,np.newaxis] + .62*np.random.rand(dlen,1))[:,0]

Data3 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+2)
Data3.values[:,1] = (.5*Data3.values[:,0][:,np.newaxis] + 1*np.random.rand(dlen,1))[:,0]


Data4 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+3.5)
Data4.values[:,1] = (-.1*Data4.values[:,0][:,np.newaxis] + .5*np.random.rand(dlen,1))[:,0]


Data1 = np.concatenate((Data1,Data2,Data3,Data4))

fig = plt.figure()
plt.plot(Data1[:,0],Data1[:,1],'ob',alpha=0.2, markersize=4)
fig.set_size_inches(7,7)

mapsize = [20,20]
som = sompy.SOMFactory.build(Data1, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')  # this will use the default parameters, but i can change the initialization and neighborhood methods
som.train(n_job=1, verbose='info')  # verbose='debug' will print more, and verbose=None wont print anything


v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)  
# could be done in a one-liner: sompy.mapview.View2DPacked(300, 300, 'test').show(som)
v.show(som, what='codebook', which_dim=[0,1], cmap=None, col_sz=6) #which_dim='all' default
# v.save('2d_packed_test')

som.component_names = ['1','2']
v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)  
v.show(som, what='codebook', which_dim='all', cmap='jet', col_sz=6) #which_dim='all' default




u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)

#This is the Umat value
UMAT  = u.build_u_matrix(som, distance=1, row_normalized=False)

#Here you have Umatrix plus its render
UMAT = u.show(som, distance2=1, row_normalized=False, show_data=True, contooor=True, blob=False)
  



