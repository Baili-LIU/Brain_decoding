# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 22:07:53 2019

@author: LIUBAILI
"""


import numpy as np
from scipy.io import loadmat
import os
 
os.environ['KERAS_BACKEND']='tensorflow'   # 调整keras的backend为tensorflow

import matplotlib.pyplot as plt
import keras
import pickle as pk
from dann_helper import DANN
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import pywt
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 


def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5): 
    #对数据预处理
    print ("Applying the desired time window.")
    
    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
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

    print ("DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain")
    print  
    subjects_train = range(1, 17)
    
    
    print ("Training on subjects", subjects_train) 

    #选取第0秒至第0.5秒的数据
    tmin = 0.0
    tmax = 0.500
    print ("Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax))

    X_train = []
    XD_train=[]
    y_train = []
    X_test = []
    y_test=[]
    ids_test = []


    '''
    载入训练数据
    '''
    print ("Creating the trainset.")
    for subject in subjects_train:
        filename = 'E:/CXSJ/data/train_subject%02d.mat' % subject
        print ("Loading", filename)
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        #XD = np.linspace(subject-1,subject-1,XX.shape[0],dtype=int)
        yy = data['y']
        sfreq = 250
        tmin_original = data['tmin']
        print ("Dataset summary:")
        print ("XX:", XX.shape)
        print ("yy:", yy.shape)
        print ("sfreq:", sfreq)

        XX = create_features(XX, tmin, tmax, sfreq)

        X_train.append(XX)
        #XD_train.append(XD)
        y_train.append(yy)

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
#    y_train = keras.utils.to_categorical(y_train, num_classes=None, dtype='int32')
    print ("Trainset:", X_train.shape)


    '''
    载入测试数据
    '''
    print ("Creating the testset.")
    subjects_test = range(17, 24)
    for subject in subjects_test:
        filename = 'E:/CXSJ/data/test_subject%02d.mat' % subject
        print ("Loading", filename)
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
 #       yy = data['y']
        #XD = np.linspace(subject-1,subject-1,num=XX.shape[0],dtype=int)
        ids = data['Id']
        sfreq = 250
        tmin_original = data['tmin']
        print ("Dataset summary:")
        print ("XX:", XX.shape)
        print ("ids:", ids.shape)
        print ("sfreq:", sfreq)

        XX = create_features(XX, tmin, tmax, sfreq)

        XD_train.append(XX)
#        y_test.append(yy)
        #XD_train.append(XD)               
        ids_test.append(ids)
        

    XD_train = np.vstack(XD_train)
   # XD_train=np.concatenate(XD_train)
    ids_test = np.concatenate(ids_test)
#    y_test = np.concatenate(y_test)
#    y_test = keras.utils.to_categorical(y_test, num_classes=None, dtype='int32')
    
   # print ("Testset:", X_test.shape)
    print ("XD_train:",XD_train.shape)
 
    
    # Initiate the model
    dann = DANN(summary=True, width=144, height=63, classes=1, features=256,grl='auto', batch_size=256, model_plot=True)
    # Train the model
    m=dann.train(X_train, XD_train, y_train, epochs=100, batch_size=512,save_model=None) #, plot_intervals=20
    plt.plot(m[0])
    plt.plot(m[1])
    y_pred=dann.model.predict(XD_train)
    
    Cls_pred=y_pred[1]
    Dom_pred=y_pred[0]
    
    filename_submission = "submission.csv"
    print ("Creating submission file", filename_submission)
    f = open(filename_submission, "w") #创建submission文件   

    print("Id,Prediction",file=f)
    Cls_result=np.zeros(len(Cls_pred),dtype=int)
    for i in range(len(Cls_pred)):
        if Cls_pred[i]<0.5:        #预测概率小于0.5，则结果为0
            Cls_result[i]=0
        else :                   #预测概率大于0.5则结果为1
            Cls_result[i]=1

        print((str(ids_test[i]) + "," + str(Cls_result[i])),file=f) #将结果写入Submission文件
    f.close()
    print('Done.')
    '''
    搭建RNN model
    '''
    '''    
    TIME_STEPS=306
    INPUT_SIZE=125
    CELL_SIZE=50
    

    model=Sequential()              #序贯模型
    model.add(SimpleRNN(            #加入RNN层
            batch_input_shape=(None, TIME_STEPS,INPUT_SIZE), #参数根据数据X_train的shape设定
            output_dim=CELL_SIZE 
            ))
    
    model.add(Dense(1))         #二分类所以参数为1
    model.add(Activation('sigmoid'))
    
    model.compile(optimizer=Adam(),loss=('binary_crossentropy'),metrics = ['accuracy'])
            
    print ("Classifier:")

    print (model)
    print ("Training.")

    model.fit(X_train, y_train,epochs=17,batch_size=512) #训练模型
    print ("Predicting.")
    
    y_pred = model.predict(X_test)   #预测结果

    
    
    
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
    
