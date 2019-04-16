# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:09:50 2017

@author: rebli
"""

import itertools
import numpy as np
import os
import matplotlib.image as mpimg       # reading images to numpy arrays

import matplotlib.pyplot as plt        # to plot any graph

from skimage import measure            # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality

from pylab import rcParams
import pandas as pd
from sklearn.model_selection import train_test_split
import Principal as principal
reload(principal)


def find_best_split_point1(histogram):
    
    histogram_values = list(itertools.chain.from_iterable(list(histogram.values())))
    prior_entropy = calculate_dict_entropy(histogram_values)
    best_distance, max_ig = 0, 0
    best_left, best_right = None, None
    for distance in histogram:
        data_left = []
        data_right = []
        for distance2 in histogram:
            if distance2 <= distance: data_left.extend(histogram[distance2])
            else: data_right.extend(histogram[distance2])
        ig = prior_entropy - (float(len(data_left))/float(len(histogram_values))*calculate_dict_entropy(data_left) + \
             float(len(data_right))/float(len(histogram_values)) * calculate_dict_entropy(data_right))
        if ig > max_ig: best_distance, max_ig, best_left, best_right = distance, ig, data_left, data_right
    return max_ig


def find_shapelets_bf1(data, max_len=100, min_len=1, plot=True, verbose=True):
    candidates = generate_candidates(data, max_len, min_len)
    bsf_gain, bsf_shapelet = 0, None
    if verbose: candidates_length = len(candidates)
    for idx, candidate in enumerate(candidates):
        #print("merda")
        #print("Entrei0")
        gain  = check_candidate1(data, candidate[0], bsf_gain)
        #print("Sai0")
        if verbose: print(idx, '/', candidates_length, ":", gain)
        if gain > bsf_gain:
            bsf_gain, bsf_shapelet = gain, candidate[0]
            
            if verbose:
                print('Found new best shapelet with gain & dist:')
            if plot:
                plt.plot(bsf_shapelet)
                plt.show()
            plt.show()
    return bsf_shapelet



def EntropyEarlyPrune(bsf_gain, dist_hist, C_A, C_B):
    minend = 0
    maxend = sorted(dist_hist.keys())[-1] + 1
    pred_dist_hist = dist_hist
    pred_dist_hist[minend] = C_A
    pred_dist_hist[maxend] = C_B
    if find_best_split_point1(pred_dist_hist) > bsf_gain:
        #print("Caralho")
        return False
    pred_dist_hist[minend] = C_B
    pred_dist_hist[maxend] = C_A
    if find_best_split_point1(pred_dist_hist) > bsf_gain:
        #print("Caralho")
        return False
    #print("vai retornar true")
    return True


def seila(histogram, data):
    siz = len(histogram)
    C_A = []
    C_B = []
    for i in data[siz:]:
        if i[1] == 1:
            C_A.append(i[0])
        else:
            C_B.append(i[0])
    return C_A, C_B        

def check_candidate1(data, shapelet, bsf_gain):
    histogram = {} 
    i = 0
    counter = 0
    #print("Entrei")
    #print(len(data))
    for entry in data:
        #print(i)
        #i += 1
        # TODO: entropy pre-pruning in each iteration
        
        time_serie, label = entry[0], entry[1]
        d, idx = subsequence_dist(time_serie, shapelet)
       
        if d is not None:
            if d not in histogram:
               histogram[d] = [(time_serie,label)]
            else:
               histogram[d].append((time_serie,label))
               
            
        counter += 1
        if counter >=20:
           #print("Entrei!!!!!!!!!!!!!!!!!1") 
           C_A, C_B = seila(histogram,data) 
           if EntropyEarlyPrune(bsf_gain, histogram, C_A, C_B):
              print "pruned"
              return 0
          
            
    return find_best_split_point1(histogram)






def draw_leaf(image):
    img = mpimg.imread(image)
    cy, cx = ndi.center_of_mass(img)
    return img, (cx, cy)


def get_contour(img, thresh=.8):
    contours = measure.find_contours(img, thresh)
    return max(contours, key=len)  # Take longest one



def convert_to_1d(file, sample=250, thresh=.8, plot=False, norm=True):
    img, (cx, cy) = draw_leaf(file)
    contour = get_contour(img, thresh)
    distances = [manhattan_distance([cx, cy], [contour[i][0], contour[i][1]]) for i in range(0,len(contour),sample)]
    distances.extend(distances)
    if plot:
        f, axarr = plt.subplots(2, sharex=False) # , sharex=True
        axarr[0].imshow(img, cmap='Set3')
        axarr[0].plot(contour[::, 1], contour[::, 0], linewidth=0.5)
        axarr[0].scatter(cx, cy)
        axarr[1].plot(distances)
        plt.show()
    if norm:
        return np.divide(distances, max(distances))
    else:
        return distances  #  Extend it twice so that it is cyclic
        
        
def generate_candidates(data, max_len=5, min_len=2):
    candidates, l = [], max_len
    while l >= min_len:
        for i in range(len(data)):
            time_serie, label = data[i][0], data[i][1]
            for k in range(len(time_serie)-l+1): candidates.append((time_serie[k:k+l], label))
        l -= 1
    return candidates


def check_candidate(data, shapelet):
    histogram = {} 
    i = 0
    for entry in data:
        # TODO: entropy pre-pruning in each iteration
        
        time_serie, label = entry[0], entry[1]
        d, idx = subsequence_dist(time_serie, shapelet)
       
        if d is not None:
            if d not in histogram:
               histogram[d] = [(time_serie,label)]
            else:
               histogram[d].append((time_serie,label))
            #histogram[d] = [(time_serie, label)] if d not in histogram else histogram[d].append((time_serie, label))
    return find_best_split_point(histogram)



def calculate_dict_entropy(data):
    counts = {}
    for entry in data:
        if entry[1] in counts: counts[entry[1]] += 1
        else: counts[entry[1]] = 1
    return calculate_entropy(np.divide(list(counts.values()), float(sum(list(counts.values())))))


def find_best_split_point(histogram):
    
    histogram_values = list(itertools.chain.from_iterable(list(histogram.values())))
    prior_entropy = calculate_dict_entropy(histogram_values)
    best_distance, max_ig = 0, 0
    best_left, best_right = None, None
    for distance in histogram:
        data_left = []
        data_right = []
        for distance2 in histogram:
            if distance2 <= distance: data_left.extend(histogram[distance2])
            else: data_right.extend(histogram[distance2])
        ig = prior_entropy - (float(len(data_left))/float(len(histogram_values))*calculate_dict_entropy(data_left) + \
             float(len(data_right))/float(len(histogram_values)) * calculate_dict_entropy(data_right))
        if ig > max_ig: best_distance, max_ig, best_left, best_right = distance, ig, data_left, data_right
    return max_ig, best_distance, best_left, best_right


def manhattan_distance(a, b, min_dist=float('inf')):
    dist = 0
    for x, y in zip(a, b):
        dist += np.abs(float(x)-float(y))
        if dist >= min_dist: return None
    return dist


def calculate_entropy(probabilities):
    return sum([-prob * np.log(prob)/np.log(2) if prob != 0 else 0 for prob in probabilities])


def subsequence_dist(time_serie, sub_serie):
    if len(sub_serie) < len(time_serie):
        min_dist, min_idx = float("inf"), 0
        for i in range(len(time_serie)-len(sub_serie)+1):
            dist = manhattan_distance(sub_serie, time_serie[i:i+len(sub_serie)], min_dist)
            if dist is not None and dist < min_dist: min_dist, min_idx = dist, i
        return min_dist, min_idx
    else:
        return None, None


def find_shapelets_bf(data, max_len=100, min_len=1, plot=True, verbose=True):
    candidates = generate_candidates(data, max_len, min_len)
    bsf_gain, bsf_shapelet = 0, None
    if verbose: candidates_length = len(candidates)
    for idx, candidate in enumerate(candidates):
        #print("merda")
        gain, dist, data_left, data_right = check_candidate(data, candidate[0])
        if verbose: print(idx, '/', candidates_length, ":", gain, dist)
        if gain > bsf_gain:
            bsf_gain, bsf_shapelet = gain, candidate[0]
            
            if verbose:
                print('Found new best shapelet with gain & dist:', bsf_gain, dist, [x[1] for x in data_left], \
                                                                                   [x[1] for x in data_right])
            if plot:
                plt.plot(bsf_shapelet)
                plt.show()
            plt.show()
    return bsf_shapelet


# min_len = 13, max_len=15

def extract_shapelets(data, min_len=150, max_len=150, verbose=1):
    _classes = np.unique([x[1] for x in data])
    shapelet_dict = {}
    for _class in _classes:
        print('Extracting shapelets for', _class)
        transformed_data = []
        for entry in data:
            time_serie, label = entry[0], entry[1]
            if label == _class: transformed_data.append((time_serie, 1))
            else: transformed_data.append((time_serie, 0))
        shapelet_dict[_class] = find_shapelets_bf(transformed_data, max_len=max_len, min_len=min_len, plot=0, verbose=1)
    return shapelet_dict





leaf_img = [('Acer Palmatum', [27, 118, 203, 324, 960, 1041,1085,1088,1370,1551]), 
            ('Acer Pictum', [146, 311, 362, 810, 915, 949, 956, 1417,1538,1578]),
            ('Quercus Coccinea', [163, 189, 469, 510, 576, 605,841,1007,1530,1539]),
            ('Quercus Rhysophylla', [375, 481, 876, 1120, 1163, 1323,1337,1399,1403,1552]),
            ('Salix Fragilis', [15, 620, 704, 847, 976, 1025,1073,1077,1472,1543])]
leaf_map = {'Acer Palmatum': 0, 'Acer Pictum': 1, 'Salix Fragilis': 2,
            'Quercus Rhysophylla': 3, 'Quercus Coccinea': 4}
data = []


data1 = pd.read_csv("dat_final132.txt", header=None)
data1['label'] = 0
j = 1     
for i in range(0,data1.shape[0]):
    data1.loc[i,'label'] = j
    j = j + 1
    if j == 10:
       j = 1
         

label = data1['label']
data1 = data1.drop('label',axis=1)
data1 = np.asarray(data1)
label = np.asarray(label)
X_train, X_test, y_train, y_test = train_test_split(data1, label, test_size=0.30, random_state=42)

for i in range(0,X_train.shape[0]):
      
      data.append((X_train[i],y_train[i]))



'''data10 = []
for img in leaf_img:
    name, image_numbers = img[0], img[1]
    for number in image_numbers:
        data10.append((convert_to_1d('images/'+str(number)+'.jpg', plot=True), 
                     leaf_map[name]))'''
        



    


shapelet_dict5 = extract_shapelets(data)



preds = []
for i in range(0,len(X_test)):
    dic = {}
    
    _dist1, _idx1 = subsequence_dist(X_test[i], shapelet_dict4[1])

    _dist2, _idx2 = subsequence_dist(X_test[i], shapelet_dict4[2])

    _dist3, _idx3 = subsequence_dist(X_test[i], shapelet_dict4[3])

    _dist4, _idx4 = subsequence_dist(X_test[i], shapelet_dict4[4])

    _dist5, _idx5 = subsequence_dist(X_test[i], shapelet_dict4[5])
    
    _dist6, _idx6 = subsequence_dist(X_test[i], shapelet_dict4[6])
    
    _dist7, _idx7 = subsequence_dist(X_test[i], shapelet_dict4[7])
    
    _dist8, _idx8 = subsequence_dist(X_test[i], shapelet_dict4[8])
    
    _dist9, _idx9 = subsequence_dist(X_test[i], shapelet_dict4[9])

    dic[1] = _dist1 
    dic[2] = _dist2
    dic[3] = _dist3
    dic[4] = _dist4
    dic[5] = _dist5
    dic[6] = _dist6
    dic[7] = _dist7
    dic[8] = _dist8
    dic[9] = _dist9   
    dic = sorted(dic.items(), key=lambda x:x[1])
    preds.append(dic[0][0])



'''cnf_matrix = confusion_matrix(y_test, preds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = ['falha_13','falha_14','falha_15']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()'''

       
for i in range(0,len(X_train)):
  print("Figura:")
  print(i)
  plt.plot(list(range(len(X_train[i]))), X_train[i])
  _dist, _idx = subsequence_dist(X_train[i], shapelet_dict3[3])
  plt.plot(list(range(_idx, _idx+len(shapelet_dict3[3]))), shapelet_dict3[3], color='r')
  plt.show()

  plt.plot(list(range(len(X_train[i]))), X_train[i])
  _dist, _idx = subsequence_dist(X_train[i], shapelet_dict3[4])
  plt.plot(list(range(_idx, _idx+len(shapelet_dict3[4]))), shapelet_dict3[4], color='r')
  plt.show()

  plt.plot(list(range(len(X_train[i]))), X_train[i])
  _dist, _idx = subsequence_dist(X_train[i], shapelet_dict3[5])
  plt.plot(list(range(_idx, _idx+len(shapelet_dict3[5]))), shapelet_dict[5], color='r')
  plt.show()


'''distances1 = convert_to_1d('images/27.jpg', plot=0, norm=1)
distances2 = convert_to_1d('images/146.jpg', plot=0, norm=1)
distances3 = convert_to_1d('images/375.jpg', plot=0, norm=1)
distances4 = convert_to_1d('images/15.jpg', plot=0, norm=1)
distances5 = convert_to_1d('images/163.jpg', plot=0, norm=1)'''

                          
'''for j in range(0,100):
                          
   #f, axarr = plt.subplots(2, sharex=True) # , sharex=True]

   plt.plot(list(range(len(data1.iloc[j,:]))), data1.iloc[j,:])
   _dist, _idx = subsequence_dist(data1.iloc[j,:], shapelet_dict[3])
   plt.plot(list(range(_idx, _idx+len(shapelet_dict[3]))), shapelet_dict[3], color='r')
   plt.show()



                
axarr[1].plot(list(range(len(data1.iloc[120,:]))), data1.iloc[120,:])
_dist, _idx = subsequence_dist(data1.iloc[120,:], shapelet_dict[4])
axarr[1].plot(list(range(_idx, _idx+len(shapelet_dict[4]))), shapelet_dict[4], color='r')

                       
axarr[2].plot(list(range(len(data1.iloc[220,:]))), data1.iloc[220,:])
_dist, _idx = subsequence_dist(data1.iloc[220,:], shapelet_dict[5])
axarr[2].plot(list(range(_idx, _idx+len(shapelet_dict[5]))), shapelet_dict[5], color='r')'''

                       





'''distances1 = convert_to_1d('images/27.jpg', plot=0, norm=1)
distances2 = convert_to_1d('images/146.jpg', plot=0, norm=1)
distances3 = convert_to_1d('images/375.jpg', plot=0, norm=1)
distances4 = convert_to_1d('images/15.jpg', plot=0, norm=1)
distances5 = convert_to_1d('images/163.jpg', plot=0, norm=1)       
                       
f, axarr = plt.subplots(5, sharex=True) # , sharex=True]                       
axarr[0].plot(list(range(len(distances1))), distances1)
_dist, _idx = subsequence_dist(distances1, shapelet_dict[0])
axarr[0].plot(list(range(_idx, _idx+len(shapelet_dict[0]))), shapelet_dict[0], color='r')

axarr[1].plot(distances2)
_dist, _idx = subsequence_dist(distances2, shapelet_dict[1])
axarr[1].plot(list(range(_idx, _idx+len(shapelet_dict[1]))), shapelet_dict[1], color='r')

axarr[2].plot(distances3)
_dist, _idx = subsequence_dist(distances3, shapelet_dict[2])
axarr[2].plot(list(range(_idx, _idx+len(shapelet_dict[2]))), shapelet_dict[2], color='r')

axarr[3].plot(distances4)
_dist, _idx = subsequence_dist(distances4, shapelet_dict[3])
axarr[3].plot(list(range(_idx, _idx+len(shapelet_dict[3]))), shapelet_dict[3], color='r')

axarr[4].plot(distances5)
_dist, _idx = subsequence_dist(distances5, shapelet_dict[4])
axarr[4].plot(list(range(_idx, _idx+len(shapelet_dict[4]))), shapelet_dict[4], color='r')

plt.show()'''



    
    
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
               
            





            
plt.plot(list(range(len(X_train[10]))), X_train[10])
_dist, _idx = subsequence_dist(X_train[10], shapelet_dict4[1])
plt.plot(list(range(_idx, _idx+len(shapelet_dict4[1]))), shapelet_dict4[1], color='r')
plt.show()

plt.plot(list(range(len(X_train[0]))), X_train[0])
_dist, _idx = subsequence_dist(X_train[0], shapelet_dict4[2])
plt.plot(list(range(_idx, _idx+len(shapelet_dict4[2]))), shapelet_dict4[2], color='r')
plt.show()

plt.plot(list(range(len(X_train[11]))), X_train[11])
_dist, _idx = subsequence_dist(X_train[11], shapelet_dict4[3])
plt.plot(list(range(_idx, _idx+len(shapelet_dict4[3]))), shapelet_dict4[3], color='r')
plt.show()

plt.plot(list(range(len(X_train[4]))), X_train[4])
_dist, _idx = subsequence_dist(X_train[4], shapelet_dict4[4])
plt.plot(list(range(_idx, _idx+len(shapelet_dict4[4]))), shapelet_dict4[4], color='r')
plt.show()



plt.plot(list(range(len(X_train[1]))), X_train[1])
_dist, _idx = subsequence_dist(X_train[1], shapelet_dict4[5])
plt.plot(list(range(_idx, _idx+len(shapelet_dict4[5]))), shapelet_dict4[5], color='r')
plt.show()

plt.plot(list(range(len(X_train[13]))), X_train[13])
_dist, _idx = subsequence_dist(X_train[13], shapelet_dict4[6])
plt.plot(list(range(_idx, _idx+len(shapelet_dict4[6]))), shapelet_dict4[6], color='r')
plt.show()



plt.plot(list(range(len(X_train[17]))), X_train[17])
_dist, _idx = subsequence_dist(X_train[17], shapelet_dict4[7])
plt.plot(list(range(_idx, _idx+len(shapelet_dict4[7]))), shapelet_dict4[7], color='r')
plt.show() 


plt.plot(list(range(len(X_train[27]))), X_train[27])
_dist, _idx = subsequence_dist(X_train[27], shapelet_dict4[8])
plt.plot(list(range(_idx, _idx+len(shapelet_dict4[8]))), shapelet_dict4[8], color='r')
plt.show()    


plt.plot(list(range(len(X_train[2]))), X_train[2])
_dist, _idx = subsequence_dist(X_train[2], shapelet_dict4[9])
plt.plot(list(range(_idx, _idx+len(shapelet_dict4[9]))), shapelet_dict4[9], color='r')
plt.show()






'''for i in range(0,len(time_serie)-len(sub_serie)+1,4):
            print(time_serie[i:i+len(sub_serie)])'''
            #dist = manhattan_distance(sub_serie, time_serie[i:i+len(sub_serie)], min_dist)
              
            

from xgboost.sklearn import XGBClassifier
import xgboost as xgb            
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#data = []


data1 = pd.read_csv("dat_final13_original.txt", header=None)
#data1 = data1 / data1.max(axis=0) 

data1['falha'] = 0
     
data2 = pd.read_csv("dat_final14_original.txt", header=None)
#data1 = data1 / data1.max(axis=0) 

data2['falha'] = 1     
     
data3 = pd.read_csv("dat_final15_original.txt", header=None)
#data1 = data1 / data1.max(axis=0) 

data3['falha'] = 2

dat_final = pd.concat([data1,data2,data3])  

dat_final = dat_final.iloc[np.random.permutation(len(dat_final))]
    
label = dat_final['falha']
dat_final = dat_final.drop('falha',axis=1)

nomes = []
for i in range(1,dat_final.shape[1]+1):
    nomes.append("variavel" + str(i))
    
dat_final.columns = nomes
#feat = dat_final.columns 

dat_var = dat_final[lista]

#-----------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X = dat_var.copy()
y = label
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=42)
        
for name, clf in zip(names, classifiers):
       
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(name)
        print(score)
        

   


        





#--------------------------------------------------------------------

xgb_params = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'min_child_weight': 1,
    'objective': 'multi:softmax',
    'eval_metric': 'merror',
    'num_class' : 3,
    'silent': 1,
     }
    
  
nomes_colunas = dat_var.columns
    
dtrain = xgb.DMatrix(dat_var,label, feature_names=nomes_colunas)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20)
num_boost_rounds = len(cv_output)

X_train, X_test, y_train, y_test = train_test_split(dat_var, label, test_size=0.30, random_state=42)
#print(type(X_train))
#print(nomes_colunas)
dtrain1 = xgb.DMatrix(X_train,y_train , feature_names=nomes_colunas)
dteste = xgb.DMatrix(X_test, feature_names=nomes_colunas)

model = xgb.train(dict(xgb_params, silent=0), dtrain1, num_boost_round=num_boost_rounds)
accu = accuracy_score(y_test,model.predict(dteste))
    
'''j = 0     
for i in range(0,data1.shape[0]):
    data1.loc[i,'label'] = str(j) + 'F3' 
    j = j + 1
    if j == 9:
       j = 0'''
         


'''data13_temp = data13[data13.label == '0F3']
data13_temp1 = data14[data14.label == '0F4']
data13_temp2 = data15[data15.label == '0F5']

data_var1 = pd.concat([data13_temp,data13_temp1,data13_temp2])


data13_temp = data13[data13.label == '1F3']
data13_temp1 = data14[data14.label == '1F4']
data13_temp2 = data15[data15.label == '1F5']

data_var2 = pd.concat([data13_temp,data13_temp1,data13_temp2])


data13_temp = data13[data13.label == '2F3']
data13_temp1 = data14[data14.label == '2F4']
data13_temp2 = data15[data15.label == '2F5']

data_var3 = pd.concat([data13_temp,data13_temp1,data13_temp2])

data13_temp = data13[data13.label == '3F3']
data13_temp1 = data14[data14.label == '3F4']
data13_temp2 = data15[data15.label == '3F5']

data_var4 = pd.concat([data13_temp,data13_temp1,data13_temp2])

data13_temp = data13[data13.label == '4F3']
data13_temp1 = data14[data14.label == '4F4']
data13_temp2 = data15[data15.label == '4F5']

data_var5 = pd.concat([data13_temp,data13_temp1,data13_temp2])

data13_temp = data13[data13.label == '5F3']
data13_temp1 = data14[data14.label == '5F4']
data13_temp2 = data15[data15.label == '5F5']

data_var6 = pd.concat([data13_temp,data13_temp1,data13_temp2])

data13_temp = data13[data13.label == '6F3']
data13_temp1 = data14[data14.label == '6F4']
data13_temp2 = data15[data15.label == '6F5']

data_var7 = pd.concat([data13_temp,data13_temp1,data13_temp2])

data13_temp = data13[data13.label == '7F3']
data13_temp1 = data14[data14.label == '7F4']
data13_temp2 = data15[data15.label == '7F5']

data_var8 = pd.concat([data13_temp,data13_temp1,data13_temp2])


data13_temp = data13[data13.label == '8F3']
data13_temp1 = data14[data14.label == '8F4']
data13_temp2 = data15[data15.label == '8F5']

data_var9 = pd.concat([data13_temp,data13_temp1,data13_temp2])'''

                     
                     
from sklearn import preprocessing


'''lbl = preprocessing.LabelEncoder()
lbl.fit(list(data_var9['label'].values)) 
data_var9['label'] = lbl.transform(list(data_var9['label'].values))'''


data1 = data_var1.iloc[:,100::].copy()
label = data1['label']
data1 = data1.drop('label',axis=1)
nomes = []
for i in range(0,data1.shape[1]):
    nomes.append("name" + str(i))
    
data1.columns = nomes
feat = data1.columns    
data1 = np.asarray(data1)
label = np.asarray(label)
#data1 = preprocessing.scale(data1)
data1 = data1 / data1.max(axis=0) 
X_train, X_test, y_train, y_test = train_test_split(data1, label, test_size=0.30, random_state=42)


xgb_params = {
    'eta': 0.05,
    #'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'min_child_weight': 1,
    'objective': 'multi:softmax',
    'eval_metric': 'merror',
    'num_class' : 3,
    'silent': 1,
     }

dtrain = xgb.DMatrix(X_train, y_train, feature_names=feat)
dtest = xgb.DMatrix(X_test,feature_names=feat)




cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50,show_stdv=False)

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

ig, ax = plt.subplots(1, 1, figsize=(6, 8))
xgb.plot_importance(model, max_num_features=90, height=0.5, ax=ax)

#dtrain1 = xgb.DMatrix(X_train)
#dtrain1 = model.predict(dtrain1)
dtest1 = model.predict(dtest)

'''print "\nModel Report"
print "Accuracy Treino: %.4g" % accuracy_score(y_train, dtrain1)'''
    
print "\nModel Report"
print "Accuracy Teste: %.4g" % accuracy_score(y_test, dtest1)                  
                  
                  

def ctun_model_dm(data,y_train):
    print("Iniciando o tunning.....")
    dat = xgb.DMatrix(data,y_train)
    
    param_grid = {
    'max_depth':range(3,10,1),
    'min_child_weight':range(1,9,1)
                 }
    
    cvs = {}
    list1 = param_grid['max_depth']
    list2 = param_grid['min_child_weight']
    c = list(itertools.product(list1, list2))
    for i in range(0,len(c)):
        print("Modelo",i)
        
        
        model = {'eta':0.05, 'subsample': 0.7, 'colsample_bytree': 0.7, 
             'objective': 'multi:softmax', 'eval_metric': 'merror' , 'silent':1, 'max_depth':c[i][0],
               'min_child_weight':c[i][1],'num_class' : 3} 
     
        cv_xgb = xgb.cv(model,dat, num_boost_round = 1000, 
                early_stopping_rounds = 20) 
         
        cvs[c[i]] = cv_xgb
    return cvs

def ctun_model_gama(data,y_train):
    print("Iniciando o tunning gamma.....")
    dat = xgb.DMatrix(data,y_train)
    
    param_grid = {
    'gamma':[i/10.0 for i in range(0,5)]
    
                 }
    
     
    cvs = {}
    list1 = param_grid['gamma']
    #list2 = param_grid['min_child_weight']
    #c = list(itertools.product(list1, list2))
    for i in list1:
        print("Modelo",i)
        
        
        model = {'eta':0.05, 'subsample': 0.7, 'colsample_bytree': 0.7, 
             'objective': 'multi:softmax', 'eval_metric': 'merror' , 'silent':1, 'max_depth':4,
               'min_child_weight':1, 'gamma':i,'num_class' : 3} 
     
        cv_xgb = xgb.cv(model,dat, num_boost_round = 1000, 
                early_stopping_rounds = 20) 
         
        cvs[i] = cv_xgb
    return cvs


def ctun_model_colsample(data,y_train):
    print("Iniciando o colsample.....")
    dat = xgb.DMatrix(data,y_train)
    
    param_grid = {
    'subsample':[i/100.0 for i in range(50,95,5)],
    'colsample_bytree':[i/100.0 for i in range(50,95,5)]
      }
    
     
    cvs = {}
    list1 = param_grid['subsample']
    list2 = param_grid['colsample_bytree']
    c = list(itertools.product(list1, list2))
    print(len(c))
    for i in range(0,len(c)):
        print("Modelo",i)
        print("Subsample",i)
        print(c[i][0])
        print("colsample_bytree",i)
        print(c[i][1])
        
        
        model = {'eta':0.05, 'subsample': c[i][0], 'colsample_bytree': c[i][1], 
             'objective': 'multi:softmax', 'eval_metric': 'merror' , 'silent':1, 'max_depth':4,
               'min_child_weight':1, 'gamma':0,'num_class' : 3} 
     
        cv_xgb = xgb.cv(model,dat, num_boost_round = 1000, 
                early_stopping_rounds = 20) 
        print("Pontua:")
        print(cv_xgb.iloc[cv_xgb.shape[0]-1]['test-merror-mean'])
        cvs[c[i]] = cv_xgb.iloc[cv_xgb.shape[0]-1]['test-merror-mean']
    return cvs


def ctun_model_reg(data,y_train):
    print("Iniciando o tunning gamma.....")
    dat = xgb.DMatrix(data,y_train)
    
    param_grid = {
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
    
                 }
    
     
    cvs = {}
    list1 = param_grid['reg_alpha']
    #list2 = param_grid['min_child_weight']
    #c = list(itertools.product(list1, list2))
    for i in list1:
        print("Modelo",i)
        
        
        model = {'eta':0.05, 'subsample': 0.8, 'colsample_bytree': 0.5, 
             'objective': 'multi:softmax', 'eval_metric': 'merror' , 'silent':1, 'max_depth':4,
               'min_child_weight':1, 'gamma':0,'reg_alpha':i,'num_class' : 3} 
     
        cv_xgb = xgb.cv(model,dat, num_boost_round = 1000, 
                early_stopping_rounds = 20) 
         
        cvs[i] = cv_xgb
    return cvs
       

def excluir_coluna(s):
           lista = []
           
           for i in range(0,len(s)-1):
               
               if s[i] > 0.6:
                  lista.append(i)
                  
      
           return lista


mydata = [{'Nearest Neighbors' :0.56,  'Linear SVM':0.56 , 'RBF SVM':0.56 , 'Random Forest':0.58, 'AdaBoost': 0.60, 'Naive Bayes': 0.57 },]
df = pd.DataFrame(mydata)
    
    
    
'''cv_mean = {}
chaves = cvs.keys()
for fol in chaves:
 cv_mean1 = cvs[fol]
 cv_mean[fol] = cv_mean1.iloc[cv_mean1.shape[0] -1]['test-merror-mean'] '''          


       
       
                  
'''plt.plot(resultado.contandofeatures) 
plt.xlabel("Interacoes")
plt.ylabel("numero de features")'''