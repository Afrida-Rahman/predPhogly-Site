import numpy as np
import pandas as pd 
from Bio import SeqIO

def Segment_Sequence():
    
    pid = []
    pid_pos_true = []
    seq = []
    j = 0

    for seq_record in SeqIO.parse('Benchmark_Dataset.fasta', "fasta"):
        header = seq_record.id
        contents = header.split('|')
        inner_contents_1 = contents[2].split(':')
        pid.append(contents[1])
        inner_contents_2 = inner_contents_1[1].split(',')
        if len(inner_contents_2)>1:
            for i in range(len(inner_contents_2)):
                pid_pos_true.append(contents[1] + ':' + inner_contents_2[i])
        else:    
            pid_pos_true.append(contents[1] + ':' + inner_contents_1[1])
        seq.append(seq_record.seq._data)
        
    
    
    
    pid = np.array(pid, dtype=str)
    pid = pid.reshape(pid.shape[0],-1)
    
    pid_pos_true = np.array(pid_pos_true, dtype=str)
    pid_pos_true = pid_pos_true.reshape(pid_pos_true.shape[0],-1)
    
    seq = np.array(seq, dtype=str)
    seq = seq.reshape(seq.shape[0],-1)
    
    segment_array=[]
    pid_pos_all = []
    window = 29
    frame = int((window)/2)
    
    for i in range(len(seq)):
        seq_i = seq[i][0]
        c=1
        for j in range(len(seq_i)):
            if c<=len(seq_i)-1:
                if c<frame and seq_i[c]=="K":
                    left=seq_i[:c]
                    right=seq_i[c+1:c+frame+1]
                    mid_k = seq_i[c]
                    pad_count = frame-c
                    pad="X"
                    k=0
                    for k in range(pad_count-1):
                        pad=pad+"X"
                    segment = pad+left+mid_k+right
                    segment_array.append(segment)
                    pid_pos_all.append(pid[i][0]+":"+str(c+1))
                    
                    # seg_pid_array.append(pid[i][0])
                    # seg_pos_array.append(c+1)
                    
                elif c+1+frame>len(seq_i) and seq_i[c]=="K":            
                    left=seq_i[c-frame:c]
                    right=seq_i[c+1:]
                    mid_k=seq_i[c]
                    pad_count = c+1+frame-len(seq_i)
                    pad="X"
                    k=0
                    for k in range(pad_count-1):
                        pad=pad+"X"
                    segment = left+mid_k+right+pad
                    segment_array.append(segment)
                    pid_pos_all.append(pid[i][0]+":"+str(c+1))
                    # seg_pid_array.append(pid[i][0])
                    # seg_pos_array.append(c+1)   
                    
                elif seq_i[c]=="K":
                    left=seq_i[c-frame:c]
                    right=seq_i[c+1:c+frame+1]
                    mid_k = seq_i[c]
                    segment = left+mid_k+right
                    segment_array.append(segment)
                    pid_pos_all.append(pid[i][0]+":"+str(c+1))
                    # seg_pid_array.append(pid[i][0])
                    # seg_pos_array.append(c+1)                
            c+=1
            # print(c)
    
    pid_pos_all = np.array(pid_pos_all, dtype=str)
    pid_pos_all = pid_pos_all.reshape(pid_pos_all.shape[0],-1)
    # seg_pid_array = pd.DataFrame(seg_pid_array)
    segment_array = pd.DataFrame(segment_array, columns = ["Samples"])
    # seg_pos_array = pd.DataFrame(seg_pos_array)
    

    labels = np.zeros(shape=(pid_pos_all.shape), dtype=np.int64)
    for i in range(len(pid_pos_true)):
        for j in range(len(pid_pos_all)):
            if pid_pos_true[i] == pid_pos_all[j]:
                labels[j] = 1
                break

    dataset = np.concatenate([pid_pos_all, segment_array[['Samples']], labels], axis=1)
    dataset = pd.DataFrame(dataset, columns = ["PID", "Samples", "Labels"])
    
    return dataset

# a = Segment_Sequence(29)