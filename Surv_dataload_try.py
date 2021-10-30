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
import matplotlib.pyplot as plt
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
    #%% 
    labels_ip=np.loadtxt(label_path)
    sizL=labels_ip.shape
    sizlen=len(sizL)
    #%%
    I=cv2.imread(img_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    sizI=I.shape   
    #%%
    if sizlen>1:
        for label_index in range(sizL[0]):
            xc=round(labels_ip[label_index,1]*sizI[0])
            yc=round(labels_ip[label_index,2]*sizI[1])
            
            xwidth=round(labels_ip[label_index,3]*sizI[0])
            yheight=round(labels_ip[label_index,4]*sizI[1])
            
            xmin=xc-round(xwidth*0.5)
            xmax=xc+round(xwidth*0.5)
            ymin=yc-round(yheight*0.5)
            ymax=yc+round(yheight*0.5)
            
            I1=I[xmin:xmax,ymin:ymax,:] # for the array slicing, the x and y changes to height and width
            # plt.figure(2),plt.subplot(2,2,label_index+1),plt.imshow(I1);plt.show() # Just for seeing purpose
            label_categ=int(labels_ip[label_index,0])
            label_categ_str=str(label_categ)
            categpath=os.path.join(folderpath, label_categ_str)
            if not os.path.isdir(categpath):
                os.makedirs(categpath)
            categ_img_fname=img_fname_p+'_'+label_categ_str+'_'+str(label_index)+'.jpg'
            categ_img_path=os.path.join(categpath, categ_img_fname)
            cv2.imwrite(categ_img_path, I1)
            # print(categ_img_path)
    else:
        xc=round(labels_ip[1]*sizI[0])
        yc=round(labels_ip[2]*sizI[1])
        
        xwidth=round(labels_ip[3]*sizI[0])
        yheight=round(labels_ip[4]*sizI[1])
        
        xmin=xc-round(xwidth*0.5)
        xmax=xc+round(xwidth*0.5)
        ymin=yc-round(yheight*0.5)
        ymax=yc+round(yheight*0.5)
        
        I1=I[xmin:xmax,ymin:ymax,:] # for the array slicing, the x and y changes to height and width
        # plt.figure(2),plt.subplot(1,2,label_index+1),plt.imshow(I1);plt.show() # Just for seeing purpose
        label_categ=int(labels_ip[0])
        label_categ_str=str(label_categ)
        categpath=os.path.join(folderpath, label_categ_str)
        if not os.path.isdir(categpath):
            os.makedirs(categpath)
        categ_img_fname=img_fname_p+'_'+label_categ_str+'.jpg'
        categ_img_path=os.path.join(categpath, categ_img_fname)
        cv2.imwrite(categ_img_path, I1)
    
#%%
folderpath='/home/arun/Documents/PyWSPrecision/datasets/Defects_CleanedData_YoloAnnotationFormat';
os.chdir(folderpath)
myFiles = glob.glob('*.txt')
myFilesL=len(myFiles)
for file in range(338,myFilesL):    
    preparedata(myFiles[file])
    print('Executed Successfully : %s'%file)