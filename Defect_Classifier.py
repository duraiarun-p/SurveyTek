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
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import prepare_data as dp
import pickle

if __name__ == '__main__':
    # file_name = "/home/arun/Documents/PyWSPrecision/Defect_Classification-main/Dataset/test/test_mould_2.jpg"
    file_name = '/home/arun/Documents/PyWSPrecision/datasets/cat_surveytek_1/valid/4/IMG_20210518_105546_4.jpg'
    kernel_size = 5
    theta_list = [0, 45, 90, 135, 180, 225, 270, 315]

    class_labels = np.load("defect_class_labels.npy")
    dataset = np.load("defect_dataset.npy")
    dataset_14400 = np.load("defect_dataset_14400.npy")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="path to image to classify")
    args = parser.parse_args()

    if args.image:
        file_name = args.image

    img = imread(file_name, 0)
    img = resize(img, (1200, 900))
    feature_maps = []
    for theta in theta_list:
        feature_maps.append(dp.steerableGaussian(img, theta, kernel_size))
        feature_maps.append(dp.steerableHilbert(img, theta, kernel_size))

    lda_input = np.array([], dtype=np.uint8)
    for feature_map in feature_maps:
        pooled = dp.imageMaxPool(feature_map)
        lda_input = np.append(lda_input,pooled)

    lda_input = np.resize(lda_input,(1, 14400))

    lda = LinearDiscriminantAnalysis(n_components=1)
    reduced = lda.fit(dataset_14400, class_labels).transform(lda_input)
    print(reduced)

    scaler = preprocessing.StandardScaler().fit(dataset)
    scaler.transform(reduced)

    #clf = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(dataset, class_labels)
    with open("svm_class_model.pkl", 'rb') as file:
        clf = pickle.load(file)
    print(clf.predict(reduced))
