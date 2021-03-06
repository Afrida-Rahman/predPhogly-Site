import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
import os

absolute_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..\\..\\'))
test_index = pd.read_csv('Test_Indices.csv')

predictors = ['CKSAAP_PhoglySite', 'iPGK-PseAAC', 'Phogly-PseAAC']
performance = pd.DataFrame(columns=['Sp','Sn','Prec','ACC','MCC','AUC'])

for k in predictors:
    pred_info = pd.read_excel('Predicted_and_Test_Labels.xlsx', sheet_name = k+"_Labels")
    y_test = pred_info["Y_Test"]
    y_pred = pred_info["Y_Pred"]
    
    i=1
    results = []
    y_test_fold = []
    y_pred_fold = []
    
    for fold in range(test_index.shape[1]):  
        
        test_ind = pd.DataFrame(test_index.iloc[:,fold].values).dropna()
        test_ind = np.array(test_ind, dtype=np.int64)
        test_ind=np.reshape(test_ind,(len(test_ind,)))
        
        y_test_split = y_test[test_ind]
        y_pred_f = y_pred[test_ind]
        
        y_test_fold.append(y_test_split)
        y_pred_fold.append(y_pred_f)
        
        if i % 10 == 0:
            x=0
            y_test_time = np.concatenate([y_test_fold[x] for x in range(10)])
            y_test_fold = []
            x=0
            y_pred_time = np.concatenate([y_pred_fold[x] for x in range(10)])
            y_pred_fold = []
            
            sp = sklearn.metrics.recall_score(y_true = y_test_time, y_pred = y_pred_time, pos_label =-1) 
            sn = sklearn.metrics.recall_score(y_true = y_test_time, y_pred = y_pred_time, pos_label =1) 
            precision = sklearn.metrics.precision_score(y_true = y_test_time, y_pred = y_pred_time)
            acc = accuracy_score(y_true = y_test_time, y_pred = y_pred_time)
            mcc = matthews_corrcoef(y_true = y_test_time, y_pred = y_pred_time, sample_weight=None)
            auc = sklearn.metrics.roc_auc_score(y_true = y_test_time, y_score = y_pred_time)
            results.append([sp, sn, precision, acc, mcc, auc])
        i+=1
    results = np.array(results, dtype=np.float64)
    results_mean = np.round(results.mean(axis=0),4)
    performance.loc[k] = results_mean
    
performance.to_csv('Other_Predictors_Performance.csv')