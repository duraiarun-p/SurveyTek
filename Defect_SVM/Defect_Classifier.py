#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:45:03 2022

@author: arun
"""
import argparse
import numpy as np
from cv2 import imread, resize
from sklearn import preprocessing
import os
import pickle
import math
from cv2 import imshow, waitKey, destroyAllWindows, filter2D, imread, resize
from theano.tensor.signal import pool
from theano import tensor as T
from theano import function
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#%%
def imageMaxPool(img):

    my_dmatrix = T.TensorType('uint8', (False,)*2)
    input = my_dmatrix('input')
    maxpool_shape = (30, 40)
    pool_out = pool.pool_2d(input, maxpool_shape, ignore_border=True)
    f = function([input],pool_out)

    #invals = numpy.random.RandomState(1).rand(3, 2, 5, 5)
    invals = img
    output = f(invals)
    return output


def showPicture(img):
    imshow('image',img)
    print("Showing Picture")
    waitKey(0)
    destroyAllWindows()


def centeredFilter(N):
    mask_shell = np.empty((N, N), dtype=np.object)
    num = N//2
    y = num
    x = -1 * num
    countX = 0
    countY = 0

    for j in mask_shell: # interate over rows
        for i in j: # interate over elements
            mask_shell[countX][countY] = [x,y]
            x = x + 1
            countY = countY + 1
        y = y - 1
        x = -1 * num
        countY = 0
        countX = countX + 1

    return mask_shell


def steerableGaussian(img, theta, kernel_size):
    G2A_array = np.empty((kernel_size, kernel_size))
    G2B_array = np.empty((kernel_size, kernel_size))
    G2C_array = np.empty((kernel_size, kernel_size))
    countX = 0
    countY = 0
    # Calculate G2A - in array not normalized
    for rows in centeredFilter(kernel_size):
        for XY in rows:
            G2A = 0.9213*(2*(XY[0]*XY[0])-1)*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            G2A_array[countX][countY] = G2A
            countY = countY + 1
        countY = 0
        countX = countX + 1


    G2A_array = (math.cos(theta)*math.cos(theta))*G2A_array

    R2A = filter2D(img, -1, G2A_array)

    # Reset index variables
    countX = 0
    countY = 0
    # Calculate G2B - in array not normalized
    for rows in centeredFilter(kernel_size):
        for XY in rows:
            G2B = 1.843*(XY[0]*XY[1])*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            G2B_array[countX][countY] = G2B
            countY = countY + 1
        countY = 0
        countX = countX + 1

    G2B_array = (-2*math.cos(theta)*math.sin(theta))*G2B_array

    R2B = filter2D(img, -1, G2B_array)

    # Reset index variables
    countX = 0
    countY = 0
    # Calculate G2A - in array not normalized
    for rows in centeredFilter(kernel_size):
        for XY in rows:
            G2C = 0.9213*(2*(XY[1]*XY[1])-1)*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            G2C_array[countX][countY] = G2C
            countY = countY + 1
        countY = 0
        countX = countX + 1

    G2C_array = (math.sin(theta)*math.sin(theta))*G2C_array
    R2C = filter2D(img, -1, G2C_array)
    addAB = np.add(R2A,R2B)
    F_theta = np.add(R2C,addAB)

    # showPicture(F_theta)

    return F_theta


def steerableHilbert(img,theta,kernel_size):

    H2A_array = np.empty((kernel_size,kernel_size))
    H2B_array = np.empty((kernel_size,kernel_size))
    H2C_array = np.empty((kernel_size,kernel_size))
    H2D_array = np.empty((kernel_size,kernel_size))

    countX = 0
    countY = 0

    for rows in centeredFilter(kernel_size):
        for XY in rows:
            H2A = 0.9780*((-2.254*XY[0])+(XY[0]*XY[0]*XY[0]))*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            H2A_array[countX][countY] = H2A
            countY = countY + 1
        countY = 0
        countX = countX + 1


    H2A_array = (math.cos(theta)*math.cos(theta)*math.cos(theta))*H2A_array
    R2A = filter2D(img, -1, H2A_array)
    # R2A_LC = (math.cos(theta)*math.cos(theta))*R2A

    #showPicture(R2A)

    # Reset index variables
    countX = 0
    countY = 0

    for rows in centeredFilter(kernel_size):
        for XY in rows:
            H2B = 0.9780*(-0.7515+(XY[0]*XY[0]))*XY[1]*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            H2B_array[countX][countY] = H2B
            countY = countY + 1
        countY = 0
        countX = countX + 1

    # print(H2B_array)

    H2B_array = (-3*math.cos(theta)*math.cos(theta)*math.sin(theta))*H2B_array

    R2B = filter2D(img, -1, H2B_array)

    #showPicture(R2B)

    # Reset index variables
    countX = 0
    countY = 0

    for rows in centeredFilter(kernel_size):
        for XY in rows:
            H2C = 0.9780*(-0.7515+(XY[1]*XY[1]))*XY[0]*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            H2C_array[countX][countY] = H2C
            countY = countY + 1
        countY = 0
        countX = countX + 1

    H2C_array = (3*math.cos(theta)*math.sin(theta)*math.sin(theta))*H2C_array

    R2C = filter2D(img, -1, H2C_array)
    # Reset index variables
    countX = 0
    countY = 0

    for rows in centeredFilter(kernel_size):
        for XY in rows:
            H2D = 0.9780*((-2.254*XY[1])+(XY[1]*XY[1]*XY[1]))*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            H2D_array[countX][countY] = H2D
            countY = countY + 1
        countY = 0
        countX = countX + 1

    H2D_array = (-1*math.sin(theta)*math.sin(theta)*math.sin(theta))*H2D_array
    R2D = filter2D(img, -1, H2D_array)
    addAB = np.add(R2A, R2B)
    addABC = np.add(R2C, addAB)
    F_theta = np.add(R2D, addABC)

    # showPicture(F_theta)

    return F_theta

#%%

if __name__ == '__main__':
    # file_name = "/home/arun/Documents/PyWSPrecision/Defect_Classification-main/Dataset/test/test_mould_2.jpg"
    
    Datadir=os.path.join(os.getcwd(),'TestData')
    
    files = os.listdir(Datadir)
    
    file_index=np.random.choice(len(files))
    
    file_name=files[file_index]
    
    file_name=os.path.join(Datadir,file_name)
    
    kernel_size = 5
    theta_list = [0, 45, 90, 135, 180, 225, 270, 315]

    class_labels = np.load("defect_class_labels.npy")
    dataset = np.load("defect_dataset.npy")
    dataset_14400 = np.load("defect_dataset_14400.npy")
    # classes_name = pickle.load("defect_classes.pkl")
    with open("defect_classes.pkl", 'rb') as file:
        classes_name = pickle.load(file)
        

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="path to image to classify")
    args = parser.parse_args()

    if args.image:
        file_name = args.image

    img = imread(file_name, 0)
    img = resize(img, (1200, 900))
    feature_maps = []
    for theta in theta_list:
        feature_maps.append(steerableGaussian(img, theta, kernel_size))
        feature_maps.append(steerableHilbert(img, theta, kernel_size))

    lda_input = np.array([], dtype=np.uint8)
    for feature_map in feature_maps:
        pooled = imageMaxPool(feature_map)
        lda_input = np.append(lda_input,pooled)

    lda_input = np.resize(lda_input,(1, 14400))

    lda = LinearDiscriminantAnalysis(n_components=1)
    reduced = lda.fit(dataset_14400, class_labels).transform(lda_input)
    print(reduced)

    scaler = preprocessing.StandardScaler().fit(dataset)
    scaler.transform(reduced)

    with open("svm_class_model.pkl", 'rb') as file:
        clf = pickle.load(file)
    
    score=clf.predict(reduced)
    class_predict=classes_name[int(score)]
    print('Predicted Class index: %s Class Name: %s'%(score,class_predict))
