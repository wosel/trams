#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
from glob import iglob
import os, sys, time
import pickle

import random



from utils import *


# In[2]:


np.random.seed(42)    
random.seed(42)





tram_type_dirs = ['1_New', '2_CKD_Long', '3_CKD_Short', '4_Old']
activity_dirs = ['accelerating', 'braking']

data = {}
id2label = {}

#these directories are intentionally run in the same order as in the desired output
#nevertheless we save a mapping id -> human_readable_name
for aid, act in enumerate(activity_dirs):
    for tid, tram_type in enumerate(tram_type_dirs):
        dir_id = aid * len(tram_type_dirs) + tid
        
        X, Y = convert_dir_to_numpy(dir_id, os.path.join('dataset', act, tram_type, '*.wav'))
        data[act+'_'+tram_type] = {'X': X, 'Y': Y}
        id2label[dir_id] = act+'_'+tram_type

X_neg, Y_neg = convert_dir_to_numpy(8, os.path.join('dataset', 'negative', 'checked', '*.wav'))
data['negative'] = {'X': X_neg, 'Y': Y_neg}
id2label[8] = 'negative'

pickle.dump(id2label, open('id2label.pkl', 'wb'))
    


# In[3]:


train_ratio = 0.7

X_train_list = []
Y_train_list = []
X_val_list = []
Y_val_list = []


for cls, values in data.items():
   
    print(cls)
    X = values['X']
    Y = values['Y']
    num_samples = X.shape[0]
    train_cutoff = int(num_samples*train_ratio)
    randperm = np.random.permutation(num_samples)
    X_tr = X[randperm[:train_cutoff], :]
    Y_tr = Y[randperm[:train_cutoff]]
    
    X_v = X[randperm[train_cutoff:], :]
    Y_v = Y[randperm[train_cutoff:]]
    
    X_train_list.append(X_tr)
    Y_train_list.append(Y_tr)
    X_val_list.append(X_v)
    Y_val_list.append(Y_v)
    
X_train = np.concatenate(X_train_list)
Y_train = np.concatenate(Y_train_list)


X_val = np.concatenate(X_val_list)
Y_val = np.concatenate(Y_val_list)


    


# In[4]:


#sklearn
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM, MLPClassifier
models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree (CART)', DecisionTreeClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('SVM (rbf, C=1.0)', SVC()))
models.append(('SVM_custom (poly, C=1000, degree=3)', SVC(C=1000.0, kernel='poly', degree=3, max_iter=100000)))
models.append(('Log Regression', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree (CART)', DecisionTreeClassifier()))
models.append(('Naive Bayes', GaussianNB()))

models.append(('Decision Tree (CART) maxdepth=5', DecisionTreeClassifier(max_depth=5)))
models.append(('Decision Tree (CART) maxdepth=10', DecisionTreeClassifier(max_depth=10)))
models.append(('Decision Tree (CART) maxdepth=20', DecisionTreeClassifier(max_depth=20)))


models.append(('RF', RandomForestClassifier()))
models.append(('RF n_estimators=5', RandomForestClassifier(n_estimators=5)))
models.append(('RF n_estimators=10', RandomForestClassifier(n_estimators=10)))
models.append(('RF n_estimators=20', RandomForestClassifier(n_estimators=20)))
models.append(('RF n_estimators=40', RandomForestClassifier(n_estimators=40)))


models.append(('KNN n_neighbors=13', KNeighborsClassifier(n_neighbors=13)))
models.append(('KNN n_neighbors=15', KNeighborsClassifier(n_neighbors=15)))
models.append(('KNN n_neighbors=17', KNeighborsClassifier(n_neighbors=17)))
models.append(('KNN n_neighbors=19', KNeighborsClassifier(n_neighbors=19)))
models.append(('KNN n_neighbors=11', KNeighborsClassifier(n_neighbors=11)))

models.append(('MLP 15 10 5', MLPClassifier(hidden_layer_sizes=(15, 10, 5), max_iter=10000)))
models.append(('MLP 10 5', MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=10000)))
models.append(('MLP 5', MLPClassifier(hidden_layer_sizes=(5,), max_iter=10000)))

results = []
names = []


# In[5]:


from sklearn.metrics import confusion_matrix
top_acc = 0.
for name, model in models:
    print(name)
    #kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    #randperm = np.random.permutation(X_train.shape[0])
    #model.fit(X_train[randperm[:1000]], Y_train[randperm[:1000]])

    model.fit(X_train, Y_train)

    predictions = model.predict(X_val)
    
    #print(predictions.shape)
    #print(predictions)
    correct = np.sum(predictions.astype(np.int32) == Y_val.astype(np.int32))
    
    '''
    labels = list(range(9))

    cm = confusion_matrix(Y_val, predictions, labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    #plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + [id2label[x] for x in labels], {'rotation': 75})
    ax.set_yticklabels([''] + [id2label[x] for x in labels])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    '''
    pd = pickle.dumps(model)
    #print('model size:', sys.getsizeof(pd))
    
    acc = accuracy_score(Y_val, predictions)
    if acc > top_acc:
        top_acc = acc
        top_model_name = name
        top_model_pickle = pd
        top_model = model
    print('accuracy: {}'.format(acc))
    print()
print('top model is {} with acc {}, size {}'.format(top_model_name, top_acc, sys.getsizeof(top_model_pickle)))
print('saving top model')
pickle.dump(top_model, open('best_model.pkl', 'wb'))

    
    


# In[ ]:




