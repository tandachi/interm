import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
import ELM
#reload(ELM)
from ELM import ELMRegressor
from elm_1 import ELMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def calcula_pca_visualizacao(dados,fim_normal,inicio_falha,falha):
    
  dados.ix[0:fim_normal:,18] = 0    
  dados.ix[inicio_falha:1000,18] = falha  
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

  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, projection='3d')
  plt.rcParams['legend.fontsize'] = 10   
  ax.scatter(transformed[pos,0], transformed[pos,1], transformed[pos,2],c='r',marker='o')
  ax.scatter(transformed[neg,0], transformed[neg,1], transformed[neg,2],c='b',marker='+')
  
  texto = 'Samples for falha normal and falha',str(falha),'falha comeca em',fim_normal
      
  plt.title(texto)
  ax.legend(loc='upper right')
  '''plt.xscale('symlog')
  plt.yscale('symlog')
  plt.ylabel('symlog both')'''
  plt.show()
  

def elm_whitout_kthernel(data,file_label,skf,elm):
    predicao_treino = []
    predicao_teste = []

    for train, test in skf.split(data, file_label):
        
      data_train = data[train]
      
      y_train = file_label[train]
      
      data_test = data[test]
      
      y_test = file_label[test]
      y_train = y_train.tolist()
      y_test = y_test.tolist()
      
      elm.fit(data_train,y_train)
      treino = elm.predict(data_train)
      teste = elm.predict(data_test)
      
      perc1 = accuracy_score(y_train,treino)
      perc2 = accuracy_score(y_test,teste)
      
      predicao_treino.append(perc1)
      predicao_teste.append(perc2)
    
    
        
      return np.mean(predicao_treino),np.mean(predicao_teste)
    
    


def elm_with_kernel(data,file_label,skf,elm):
    
    predicao_treino = []
    predicao_teste = []
    
    for train, test in skf.split(data, file_label):
        
      data_train = data[train]
      
      y_train = file_label[train]
      
      data_test = data[test]
      
      y_test = file_label[test]
      y_train = y_train.tolist()
      y_test = y_test.tolist()
      
      elm._local_train(data_train,y_train,False)
      treino = elm._local_test(data_train)
      treino_lista = list(map(elm.iters,treino))
      #print('teste')
      perc1 = accuracy_score(y_train,treino_lista)
      #print('teste2')
      teste = elm._local_test(data_test)
      #print('teste1')
      lista = list(map(elm.iters,teste))
      perc = accuracy_score(y_test,lista)
      predicao_treino.append(perc1)
      predicao_teste.append(perc)
      
      
    return np.mean(predicao_treino),np.mean(predicao_teste)  
    
      

#dados = pd.read_csv("X_2.csv", sep=";", header=None)
#calcula_pca_visualizacao(dados,500,501,2)

dados = pd.read_csv("X_2_1_desbalanceado.csv", sep=";", header=None)
calcula_pca_visualizacao(dados,900,901,2)

#dados = pd.read_csv("X_7.csv", sep=";", header=None)
#calcula_pca_visualizacao(dados,600,601,7)

#dados = pd.read_csv("X_6.csv", sep=";", header=None)
#calcula_pca_visualizacao(dados,300,301,6)



''''predicao_final_treino = []
predicao_final_teste = []
skf = StratifiedKFold(n_splits=10)
data = pd.read_csv("X_2_1_desbalanceado.csv", sep=";", header=None)
data.ix[0:500:,18] = 0
data.ix[501:1000,18] = 1

file_label = data.ix[:,18]
data = data.drop(18,1)
stdsc = StandardScaler()
data = stdsc.fit_transform(data)'''
'''data = data.as_matrix()'''
'''data = np.apply_along_axis(lambda x: x/np.linalg.norm(x),1,dados)'''


"""seq_1 = np.arange(100,4000,20)
for i in enumerate(seq_1):
    '''elm = ELMRegressor(n_hidden_units=i[1])'''
    elm = ELMClassifier(n_hidden=i[1], activation_func='sigmoid')
    treino,teste = elm_whitout_kthernel(data,file_label,skf,elm)
    predicao_final_treino.append(treino)
    predicao_final_teste.append(teste)  
    


plt.figure(1)
plt.subplot(211)
plt.plot(seq_1,predicao_final_treino)
plt.plot(seq_1,predicao_final_teste)
plt.title("falha 2 sem kernel - iniciando em 500")
plt.show()"""



predicao_final_treino_2 = []
predicao_final_teste_2 = []
stdsc = StandardScaler()
skf = StratifiedKFold(n_splits=10)
data = pd.read_csv("X_2_1_desbalanceado.csv", sep=";", header=None)
data.ix[0:900:,18] = 0
data.ix[901:1000,18] = 1
file_label = data.ix[:,18]
data = data.drop(18,1)
data = stdsc.fit_transform(data)
'''data = data.as_matrix()'''

seq_2 = np.arange(100,1000,20)
for i in enumerate(seq_2):
    '''elm = ELMRegressor(n_hidden_units=i[1])'''
    elm = ELMClassifier(n_hidden=i[1], activation_func='sigmoid')
    treino,teste = elm_whitout_kthernel(data,file_label,skf,elm)
    predicao_final_treino_2.append(treino)
    predicao_final_teste_2.append(teste) 
    
 

plt.figure(1)
plt.subplot(211)
plt.plot(seq_2,predicao_final_treino_2)
plt.plot(seq_2,predicao_final_teste_2)
plt.title("falha 2 sem kernel - iniciando em 900")
plt.show()

##################################################################3

predicao_final_treino_3 = []
predicao_final_teste_3 = []

data = pd.read_csv("X_2.csv", sep=";", header=None)
data.ix[0:500:,18] = 0
data.ix[501:1000,18] = 1

file_label = data.ix[:,18]
data = data.drop(18,1)
data = data.as_matrix()
'''data = stdsc.fit_transform(data)'''
param1 = ['poly',9,[4,3]]
params_2 = np.arange(-20,20)
for i in enumerate(params_2):
    param1[2] = [i[1],3]
    elm = ELMRegressor(100,param1)
    train,teste = elm_with_kernel(data,file_label,skf,elm)
    predicao_final_treino_3.append(treino)
    predicao_final_teste_3.append(teste)  
    


plt.figure(1)
plt.subplot(211)
plt.plot(params_2,[np.mean(var) for var in predicao_final_treino_3])
plt.plot(params_2,predicao_final_teste_3)
plt.title("falha 2 com kernel polinomial - iniciando em 500")
plt.show()


#############################################################3


predicao_final_treino_4 = []
predicao_final_teste_4 = []

data = pd.read_csv("X_2.csv", sep=";", header=None)
data.ix[0:500:,18] = 0
data.ix[501:1000,18] = 1

file_label = data.ix[:,18]
data = data.drop(18,1)
data = data.as_matrix()
'''data = stdsc.fit_transform(data)'''
param1 = ['rbf',9,3]
params_3 = np.arange(-20,20)
for i in enumerate(params_3):
    param1[2] = [i[1]]
    elm = ELMRegressor(100,param1)
    train,teste = elm_with_kernel(data,file_label,skf,elm)
    predicao_final_treino_4.append(train)
    predicao_final_teste_4.append(teste)  
    


plt.figure(1)
plt.subplot(211)
plt.plot(params_3,predicao_final_treino_4)
plt.plot(params_3,predicao_final_teste_4)
plt.title("falha 2 com kernel rbf - iniciando em 500")
plt.show()

#########################################################################

predicao_final_treino_5 = []
predicao_final_teste_5 = []

data = pd.read_csv("X_7.csv", sep=";", header=None)
data.ix[0:600:,18] = 0
data.ix[601:1000,18] = 1

file_label = data.ix[:,18]
data = data.drop(18,1)
data = data.as_matrix()
'''data = stdsc.fit_transform(data)'''
param1 = ['rbf',9,3]
params_4 = np.arange(-20,3)
for i in enumerate(params_4):
    param1[2] = [i[1]]
    elm = ELMRegressor(100,param1)
    train,teste = elm_with_kernel(data,file_label,skf,elm)
    predicao_final_treino_5.append(train)
    predicao_final_teste_5.append(teste) 


plt.figure(1)
plt.subplot(211)
plt.plot(params_4,predicao_final_treino_5)
plt.plot(params_4,predicao_final_teste_5)
plt.title("falha 7 com kernel rbf - iniciando em 600")
plt.show()



predicao_final_treino_6 = []
predicao_final_teste_6 = []

data = pd.read_csv("X_7.csv", sep=";", header=None)
data.ix[0:600:,18] = 0
data.ix[601:1000,18] = 1
file_label = data.ix[:,18]
data = data.drop(18,1)
data = stdsc.fit_transform(data)
'''data = data.as_matrix()'''


seq_5 = np.arange(100,4000,20)
for i in enumerate(seq_5):
    '''elm = ELMRegressor(n_hidden_units=i[1])'''
    elm = ELMClassifier(n_hidden=i[1], activation_func='sigmoid')
    treino,teste = elm_whitout_kthernel(data,file_label,skf,elm)
    predicao_final_treino_6.append(treino)
    predicao_final_teste_6.append(teste)  
    

plt.figure(1)
plt.subplot(211)
plt.plot(seq_5,predicao_final_treino_6)
plt.plot(seq_5,predicao_final_teste_6)
plt.title("falha 7 sem kernel - iniciando em 600")
plt.show()


'''----------------------------------------------------------------------'''

data = pd.read_csv("X_2.csv", sep=";", header=None)
data.ix[0:500:,18] = 0
data.ix[501:1000,18] = 1
file_label = data.ix[:,18]
data = data.drop(18,1)
'''stdsc = StandardScaler()'''
data = stdsc.fit_transform(data)
maximo = predicao_final_teste.index(max(predicao_final_teste))
class_names = ['Normal','Falha2']

X_train, X_test, y_train, y_test = train_test_split(data, file_label, test_size=0.4, random_state=0)
elm = ELMClassifier(n_hidden=seq_1[maximo], activation_func='sigmoid')
y_train = y_train.tolist()
y_test = y_test.tolist()
elm.fit(X_train,y_train)
y_pred = elm.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

'''-----------------------------------'''''

data = pd.read_csv("X_2_1_desbalanceado.csv", sep=";", header=None)
data.ix[0:900:,18] = 0
data.ix[901:1000,18] = 1
file_label = data.ix[:,18]
data = data.drop(18,1)
data = stdsc.fit_transform(data)

maximo = predicao_final_teste_2.index(max(predicao_final_teste_2))
class_names = ['Normal','Falha2_desbalancado']


X_train, X_test, y_train, y_test = train_test_split(data, file_label, test_size=0.4, random_state=0)
elm = ELMClassifier(n_hidden=seq_2[maximo], activation_func='sigmoid')
y_train = y_train.tolist()
y_test = y_test.tolist()
elm.fit(X_train,y_train)
y_pred = elm.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

'''----------------------------------------------------------'''

data = pd.read_csv("X_2.csv", sep=";", header=None)
data.ix[0:500:,18] = 0
data.ix[501:1000,18] = 1

file_label = data.ix[:,18]
data = data.drop(18,1)
data = data.as_matrix()

maximo = predicao_final_teste_3.index(max(predicao_final_teste_3))
class_names = ['Normal','Falha2_kernel_poly']


X_train, X_test, y_train, y_test = train_test_split(data, file_label, test_size=0.4, random_state=0)
param1 = ['poly',9,[4,3]]
param1[2] = [params_2[maximo],3]
elm = ELMRegressor(100,param1)

y_train = y_train.tolist()
y_test = y_test.tolist()
elm._local_train(X_train,y_train,False)
treino = elm._local_test(X_train)
treino_lista = map(elm.iters,treino)
teste = elm._local_test(X_test)
lista = map(elm.iters,teste)


cnf_matrix = confusion_matrix(y_test, lista)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

'''------------------------------------------------------------------'''

data = pd.read_csv("X_2.csv", sep=";", header=None)
data.ix[0:500:,18] = 0
data.ix[501:1000,18] = 1

file_label = data.ix[:,18]
data = data.drop(18,1)
data = data.as_matrix()

maximo = predicao_final_teste_4.index(max(predicao_final_teste_4))
class_names = ['Normal','Falha2_kernel_rbf']


X_train, X_test, y_train, y_test = train_test_split(data, file_label, test_size=0.4, random_state=0)
param1 = ['rbf',9,3]
param1[2] = [params_3[maximo],3]
elm = ELMRegressor(100,param1)

y_train = y_train.tolist()
y_test = y_test.tolist()
elm._local_train(X_train,y_train,False)
treino = elm._local_test(X_train)
treino_lista = map(elm.iters,treino)
teste = elm._local_test(X_test)
lista = map(elm.iters,teste)


cnf_matrix = confusion_matrix(y_test, lista)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

'''-------------------------------------------------------------'''


data = pd.read_csv("X_7.csv", sep=";", header=None)
data.ix[0:600:,18] = 0
data.ix[601:1000,18] = 1

file_label = data.ix[:,18]
data = data.drop(18,1)
data = data.as_matrix()
maximo = predicao_final_teste_5.index(max(predicao_final_teste_5))
class_names = ['Normal','Falha7_kernel_rbf']


X_train, X_test, y_train, y_test = train_test_split(data, file_label, test_size=0.4, random_state=0)
param1 = ['rbf',9,3]
param1[2] = [params_4[maximo],3]
elm = ELMRegressor(100,param1)

y_train = y_train.tolist()
y_test = y_test.tolist()
elm._local_train(X_train,y_train,False)
treino = elm._local_test(X_train)
treino_lista = map(elm.iters,treino)
teste = elm._local_test(X_test)
lista = map(elm.iters,teste)


cnf_matrix = confusion_matrix(y_test, lista)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

'''----------------------------------------------------------------'''
data = pd.read_csv("X_7.csv", sep=";", header=None)
data.ix[0:600:,18] = 0
data.ix[601:1000,18] = 1
file_label = data.ix[:,18]
data = data.drop(18,1)
'''stdsc = StandardScaler()'''
data = stdsc.fit_transform(data)
maximo = predicao_final_teste_6.index(max(predicao_final_teste_6))
class_names = ['Normal','Falha6']

X_train, X_test, y_train, y_test = train_test_split(data, file_label, test_size=0.4, random_state=0)
elm = ELMClassifier(n_hidden=seq_1[maximo], activation_func='sigmoid')
y_train = y_train.tolist()
y_test = y_test.tolist()
elm.fit(X_train,y_train)
y_pred = elm.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

'''------------------------------------------------------------'''

    
    
