#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:06:09 2021

@author: arun
"""
import glob
import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
#%%
def preparedata(label_fname):  
    # label_fname=myFiles[0]
    img_fname_lis=label_fname.split('.')
    img_fname_p=img_fname_lis[0]
    img_fname=img_fname_lis[0]+'.jpg'
    
    # label_fname='20200917_083150.txt'
    # img_fname='20200917_083150.jpg'
    # img_fname_p='20200917_083150'
    
    label_path=os.path.join(folderpath, label_fname)
    img_path=os.path.join(folderpath, img_fname)
    I=cv2.imread(img_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    # sizI=I.shape 
    #%% 
    labels_ip=np.loadtxt(label_path)
    sizL=labels_ip.shape
    sizlen=len(sizL)
    if sizlen>1:
        label_categ=int(labels_ip[0][0])
    else:
        label_categ=int(labels_ip[0])
    label_categ_str=str(label_categ)
    categpath=os.path.join(opfolderpath, label_categ_str)
    if not os.path.isdir(categpath):
        os.makedirs(categpath)
    categ_img_fname=img_fname_p+'_'+label_categ_str+'.jpg'
    categ_img_path=os.path.join(categpath, categ_img_fname)
    cv2.imwrite(categ_img_path, I)
    
#%%
folderpath='/home/arun/Documents/PyWSPrecision/datasets/Defects_CleanedData_YoloAnnotationFormat'
opfolderpath='/home/arun/Documents/PyWSPrecision/datasets/cat_surveytek_1/'
os.chdir(folderpath)
myFiles = glob.glob('*.txt')
myFilesL=len(myFiles)
for file in range(338,myFilesL):    
    preparedata(myFiles[file])
    print('Executed Successfully : %s'%file)

# label_fname=myFiles[0]
