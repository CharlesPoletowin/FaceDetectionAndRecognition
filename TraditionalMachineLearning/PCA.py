import os
import operator
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import time

# define PCA
def pca(data,k):
    data=np.float32(np.mat(data))
    rows,cols=data.shape
    data_mean=np.mean(data,0) # average
    data_mean_all = np.tile(data_mean,(rows,1))
    Z=data-data_mean_all
    T1=Z*Z.T
    D,V=np.linalg.eig(T1) # 特征值和特征向量
    V1=V[:,0:k]
    V1=Z.T*V1
    for i in range(k):
        L=np.linalg.norm(V1[:,i])
        V1[:,i]=V1[:,i]/L
    data_new=Z.T*V1
    return data_new,data_mean,V1


def img2vector(filename):
    # read as 'gray'
    img=cv2.imread(filename,0)
    cv2.imshow('',img)
    rows,cols=img.shape
    imgVector=np.zeros((1,rows*cols))
    imgVector=np.reshape(img,(1,cols*rows))
    return imgVector


# load dataset
def loadDataSet(k):
    dataSetDir='orl_faces'
    choose = np.random.permutation(10)+1
    train_face=np.zeros((40*k,112*92))
    train_face_number=np.zeros(40*k)
    test_face=np.zeros((40*(10-k),112*92))
    test_face_number=np.zeros(40*(10-k))
    for i in range(40):
        people_num=i+1
        for j in range(10):
            if j<k:
                filename=dataSetDir+'/s'+str(people_num)+'/'+str(choose[j])+'.pgm'
                img = img2vector(filename)
                train_face[i*k+j,:]=img
                train_face_number[i*k+j]=people_num
            else:
                filename=dataSetDir+'/s'+str(people_num)+'/'+str(choose[j])+'.pgm'
                img = img2vector(filename)
                test_face[i*k+j,:]=img
                test_face_number[i*(10-k)+(j-k)]=people_num
    return train_face,train_face_number,test_face,test_face_number


# calculate the accuracy of the test_face
def facefind():
    train_face,train_face_number,test_face,test_face_number=loadDataSet(3)
    data_train_new,data_mean,V=pca(train_face,30)
    num_train=data_train_new.shape[0]
    num_test=test_face.shape[0]
    temp_face=test_face-np.tile(data_mean,(num_test,1))
    data_test_new=temp_face*V
    data_test_new=np.array(data_test_new)
    data_train_new=np.array(data_train_new)
    true_num=0
    for i in range(num_test):
        testFace=data_test_new[i,:]
        diffMat=data_train_new-np.tile(testFace,(num_train,1))
        sqDiffMat=diffMat*2
        sqDistances=sqDiffMat.sum(axis=1)
        sortedDistIndicies=sqDistances.argsort()
        indexMin=sortedDistIndicies[0]
        if train_face_number[indexMin]==test_face_number[i]:
            true_num+=1
    
    accuracy=np.float(true_num)/num_test
    print("The classify accuracy is %.2f%%."%(accuracy*100))

if __name__ == "__main__":
    facefind()