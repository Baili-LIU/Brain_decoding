# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 23:50:41 2019

@author: LIUBAILI
"""

import numpy as np
import pywt
from scipy.io import loadmat
import os
 
os.environ['KERAS_BACKEND']='tensorflow' 

from keras.models import load_model


def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5): 
    #对数据预处理
    print ("Applying the desired time window.")
    
    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    print(beginning,end)
    XX = XX[:, 162:306, beginning:end].copy()
    #截取[tmin,tmax]的数据

    print ("Features Normalization.")
    XX -= XX.mean(0)
    XX = np.nan_to_num(XX / XX.std(0))
    #进行z-scoring数据标准化
    print("小波去噪")
    X=[]
    XF=np.empty((XX.shape[0],XX.shape[1],63))
    for trail in range(XX.shape[0]):
        for channel in range(XX.shape[1]):
            X=pywt.dwt(XX[trail][channel],'db1')
            X=X[0]
            XF[trail][channel]=X
    return XF


if __name__ == '__main__':
    
    tmin = 0.0
    tmax = 0.500
    print ("Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax))
    
    
    X_test = []
    ids_test = []
    
    print ("Creating the testset.")
    subjects_test = range(17, 18)
    for subject in subjects_test:
        filename = 'E:/CXSJ/data/test_subject%02d.mat' % subject
        print ("Loading", filename)
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        ids = data['Id']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        print ("Dataset summary:")
        print ("XX:", XX.shape)
        print ("ids:", ids.shape)
        print ("sfreq:", sfreq)

        XX = create_features(XX, tmin, tmax, sfreq)

        X_test.append(XX)
        ids_test.append(ids)

    X_test = np.vstack(X_test)
    ids_test = np.concatenate(ids_test)
    print ("Testset:", X_test.shape)
    
    
'''    
    model=load_model('SimpleRNN_model')
    
    print ("Predicting.")

    y_pred=dann.model.predict(XD_train)
    
    
    filename_submission = "submission.csv"
    print ("Creating submission file", filename_submission)
    f = open(filename_submission, "w") #创建submission文件   

    print("Id,Prediction",file=f)
    y_result=np.zeros(len(y_pred))
    for i in range(len(y_pred)):
        if y_pred[i]<0.5:        #预测概率小于0.5，则结果为0
            y_result[i]=0
        else :                   #预测概率大于0.5则结果为1
            y_result[i]=1

        print((str(ids_test[i]) + "," + str(y_result[i])),file=f) #将结果写入Submission文件
    f.close()
    print('Done.')
    
'''   
    
    