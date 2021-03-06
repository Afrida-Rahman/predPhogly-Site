import numpy as np
import pandas as pd
import sklearn.metrics

predictors = ['CKSAAP_PhoglySite', 'iPGK-PseAAC', 'Phogly-PseAAC', 'predPhogly-Site']

result = pd.DataFrame()
for i in predictors:

    pred_info = pd.read_excel('Predicted_and_Test_Labels.xlsx', sheet_name = i+'_Labels')
    y_test = np.array(pred_info['Y_Test'], dtype=np.int64)
    y_pred = np.array(pred_info['Y_Pred'], dtype=np.int64)
    
    result1=[]
    acc = sklearn.metrics.accuracy_score(y_true = y_test, y_pred = y_pred)
    mcc = sklearn.metrics.matthews_corrcoef(y_true = y_test, y_pred = y_pred, sample_weight=None)
    precision = sklearn.metrics.precision_score(y_true = y_test, y_pred = y_pred)
    sp = sklearn.metrics.recall_score(y_true = y_test, y_pred = y_pred, pos_label =-1) 
    sn = sklearn.metrics.recall_score(y_true = y_test, y_pred = y_pred, pos_label =1) 
    auc = sklearn.metrics.roc_auc_score(y_true = y_test, y_score = y_pred)
    result1.append([round(sp, 4), round(sn, 4), round(precision, 4), round(acc, 4), round(mcc, 4), round(auc, 4)])
    result1 = pd.DataFrame(result1, index = [i], columns=['Sp','Sn','Prec','ACC','MCC','AUC'])
    result = pd.concat([result, result1], axis=0)

result.to_csv('IndTestPerformance.csv', header=['Sp','Sn','Prec','ACC','MCC','AUC'])

