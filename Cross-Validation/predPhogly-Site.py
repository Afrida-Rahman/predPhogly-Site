# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import math
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import csv
import Segmentation
import Sequence_Coupling
# import os

# absolute_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..\\..\\'))

dataset = Segmentation.Segment_Sequence()

##################
X_train_positive = dataset[dataset['Labels'].isin([1])]
X_train_negative = dataset[dataset['Labels'].isin([0])]
dataset = pd.concat([X_train_positive, X_train_negative], axis=0)
###################

print("Segmentation Done!")
train_index = pd.read_csv('Train_Indices.csv')
test_index = pd.read_csv('Test_Indices.csv')


results = []
i = 1
y_test_fold=[]
y_pred_fold=[]
y_pred_score_fold=[]

wap = dataset.shape[0]/(2*X_train_positive.shape[0])
wan = dataset.shape[0]/(2*X_train_negative.shape[0])
weight = {0:wan, 1:wap}


C = np.array([math.pow(2,0),math.pow(2,0),math.pow(2,0),math.pow(2,0),math.pow(2,0),math.pow(2,1),math.pow(2,2),math.pow(2,2),math.pow(2,0),math.pow(2,0)])
gamma = np.array([math.pow(2,-1),math.pow(2,-2),math.pow(2,-2),math.pow(2,-2),math.pow(2,-2),math.pow(2,-1),math.pow(2,-2),math.pow(2,-2),math.pow(2,-2),math.pow(2,-2)])
   
C = C.reshape(C.shape[0],-1)
gamma = gamma.reshape(gamma.shape[0],-1)


time=0

for fold in train_index.columns:  
    
    train_ind = pd.DataFrame(train_index.loc[:,fold].values).dropna()
    train_ind = np.array(train_ind, dtype=np.int64)
    train_ind=np.reshape(train_ind,(len(train_ind,)))
    
    test_ind = pd.DataFrame(test_index.loc[:,fold].values).dropna()
    test_ind = np.array(test_ind, dtype=np.int64)
    test_ind=np.reshape(test_ind,(len(test_ind,)))
    
    if (i-1)%10==0:
        time=int((i-1)/10)
        print("\n\n")
        print("Iteration: "+str(time))
        
    ### Train Folds ###
    train_dataset_split = dataset.iloc[train_ind]
    X_train_split = train_dataset_split['Samples']
    y_train_split = train_dataset_split['Labels']
    y_train_split = np.array(y_train_split, dtype=np.int64)
    
    
    ### Validation Folds  ###
    test_dataset_split = dataset.iloc[test_ind]
    X_test_split = test_dataset_split['Samples']
    y_test_split = test_dataset_split['Labels']
    y_test_split = np.array(y_test_split, dtype=np.int64)
    
    
    ### Feature Extraction ###
    train_features_split = Sequence_Coupling.Extract(X_train_split)
    print("Reading Extracted Training Feature Done!")
    test_features_split = Sequence_Coupling.Extract(X_test_split)
    print("Reading Extracted Test Feature Done!")
    
    
    #### Initialize Classifier  ####  
    classifier = SVC(C=C[time], kernel='rbf', gamma=gamma[time], class_weight=weight, cache_size=500,  random_state = 0)
    classifier.fit(train_features_split, y_train_split)
    
    
    ####  Prediction  ####
    y_pred = classifier.predict(test_features_split)
    y_test_split = y_test_split.reshape(y_test_split.shape[0],-1)
    y_pred_f = y_pred.reshape(y_pred.shape[0],-1)

    y_test_fold.append(y_test_split)
    y_pred_fold.append(y_pred_f)
    print(fold)
    
    if i % 10 == 0:
        x=0
        y_test_time = np.concatenate([y_test_fold[x] for x in range(10)])
        y_test_fold = []
        x=0
        y_pred_time = np.concatenate([y_pred_fold[x] for x in range(10)])
        y_pred_fold = []
        
        sp = sklearn.metrics.recall_score(y_true = y_test_time, y_pred = y_pred_time, pos_label =0) 
        sn = sklearn.metrics.recall_score(y_true = y_test_time, y_pred = y_pred_time, pos_label =1) 
        precision = sklearn.metrics.precision_score(y_true = y_test_time, y_pred = y_pred_time)
        acc = accuracy_score(y_true = y_test_time, y_pred = y_pred_time)
        mcc = matthews_corrcoef(y_true = y_test_time, y_pred = y_pred_time, sample_weight=None)
        auc = sklearn.metrics.roc_auc_score(y_true = y_test_time, y_score = y_pred_time)
        results.append([sp, sn, precision, acc, mcc, auc])
        
    i+=1


results = np.array(results, dtype=np.float64)
results_mean = np.round(results.mean(axis=0),4).reshape(1,6)
results_std = np.round(results.std(axis=0),4).reshape(1,6)
results_mean_std = np.concatenate([results, results_mean, results_std], axis=0)
performance = pd.DataFrame(results_mean_std, index=['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5','Fold 6','Fold 7',
                                                    'Fold 8','Fold 9','Fold 10','Mean','STD'],
                                                    columns=['Sp','Sn','Prec','ACC','MCC','AUC'])
performance.to_csv("predPhogly-Site_Performance.csv")    
