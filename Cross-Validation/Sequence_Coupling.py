import numpy as np
import pandas as pd

def Extract(SegmentedSequence):
    feature_set = pd.DataFrame(SegmentedSequence)
    feature_set_list = []
    i=0
    for i in range(len(feature_set)):
        feature_set1 = list(feature_set.iloc[i,0])
        feature_set2 = np.array(feature_set1,dtype=None)
        feature_set_list.append(feature_set2)
        feature_set1=[]
        
    feature_set_list = np.array(feature_set_list, dtype=None)
    
    prob_pos_matrix = pd.read_excel('Non-Conditional_Features.xlsx', sheet_name='Non-ConditionalFeaturesPositive', index_col=0)
    prob_neg_matrix = pd.read_excel('Non-Conditional_Features.xlsx', sheet_name='Non-ConditionalFeaturesNegative', index_col=0)
    cond_prob_pos_matrix = pd.read_excel('Conditional_Features.xlsx', sheet_name='ConditionalFeaturesPositive', index_col=0, na_filter = False)
    cond_prob_neg_matrix = pd.read_excel('Conditional_Features.xlsx', sheet_name='ConditionalFeaturesNegative', index_col=0, na_filter = False)

    sample_size = len(feature_set)
    seq_len = len(feature_set.iloc[0,0])
    middle = int((seq_len-1)/2)
    coupling = np.zeros((sample_size,seq_len-1), dtype=None)
    
    for i in range(sample_size):
        feature_sample = feature_set_list[i]
        for j in range(seq_len):
            if j == middle:
                continue
            elif j == (middle-1) or j == (middle+1):  #### non-conditional
                feature = feature_sample[j]
                if j == (middle-1):
                    prob_pos = prob_pos_matrix.loc[feature+'\u207a', "Position: "+str(j+1)]  
                    prob_neg = prob_neg_matrix.loc[feature+'\u207b', "Position: "+str(j+1)]  
                    coupling[i,j] = prob_pos-prob_neg
                else:
                    prob_pos = prob_pos_matrix.loc[feature+'\u207a', "Position: "+str(j)] 
                    prob_neg = prob_neg_matrix.loc[feature+'\u207b', "Position: "+str(j)] 
                    coupling[i,j-1] = prob_pos-prob_neg
                
                
            elif j < middle or j>middle:
                if j<middle:
                #### conditional left side
                    feature1 = feature_sample[j]
                    feature2 = feature_sample[j+1]
                    feature12 = feature1+"|"+feature2
                    cond_prob_pos = cond_prob_pos_matrix.loc[feature12+'\u207a', "Position: "+str(j+1)]
                    cond_prob_neg = cond_prob_neg_matrix.loc[feature12+'\u207b', "Position: "+str(j+1)]
                    coupling[i,j] = cond_prob_pos-cond_prob_neg
                else:
                    #### conditional right side
                    feature1 = feature_sample[j]
                    feature2 = feature_sample[j-1]
                    feature12 = feature1+"|"+feature2
                    cond_prob_pos = cond_prob_pos_matrix.loc[feature12+'\u207a', "Position: "+str(j)]
                    cond_prob_neg = cond_prob_neg_matrix.loc[feature12+'\u207b', "Position: "+str(j)]
                    coupling[i,j-1] = cond_prob_pos-cond_prob_neg
                
                
                
                
    SequenceCoupling = pd.DataFrame(coupling)
    return SequenceCoupling

