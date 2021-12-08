#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 09:30:01 2021

@author: arun
SurveyTek - Without Data Augumentation using image_dataset_from_directory method
"""
# from IPython.terminal.embed import InteractiveShellEmbed
# ipshell = InteractiveShellEmbed()
# ipshell.dummymode=True
# ipshell.magic("%clear")
# ipshell.magic("%reset -f")
# ipshell.magic("%logstart -o")
import time
import datetime
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
import os
import sys
import PIL
from PIL import Image
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ""

print(device_lib.list_local_devices())

cfg = tf.compat.v1.ConfigProto() 
cfg.gpu_options.allow_growth = True
sess= tf.compat.v1.Session(config=cfg)
#%%
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
f=open('console_output.txt','w+')
os.system("nvidia-smi")
print('Init')
#%% CNN Models

# def make_VGG16_model():
#     inputs = keras.Input(shape=input_shape)
#     x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
#     x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    
    
#     outputs =layers.Dense(num_classes)(x)
#     return keras.Model(inputs, outputs)

def make_VGG16_model(input_shape,num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    
    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x=layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
    
    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(128, kernel_size=(3, 3), activation="relu")(x)
    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(128, kernel_size=(3, 3), activation="relu")(x)
    x=layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(256, kernel_size=(3, 3), activation="relu")(x)
    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(256, kernel_size=(3, 3), activation="relu")(x)
    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(256, kernel_size=(3, 3), activation="relu")(x)
    x=layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(512, kernel_size=(3, 3), activation="relu")(x)
    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(512, kernel_size=(3, 3), activation="relu")(x)
    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(512, kernel_size=(3, 3), activation="relu")(x)
    x=layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(512, kernel_size=(3, 3), activation="relu")(x)
    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(512, kernel_size=(3, 3), activation="relu")(x)
    x=layers.ZeroPadding2D(padding=(1,1))(x)
    x=layers.Conv2D(512, kernel_size=(3, 3), activation="relu")(x)
    x=layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    x=layers.Flatten()(x)
    x=layers.Dense(4096, activation='relu')(x)
    x=layers.Dropout(0.5)(x)
    x=layers.Dense(4096, activation='relu')(x)
    x=layers.Dropout(0.5)(x)
    outputs =layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

def make_simple_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    # x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
    # for size in [2, 4, 16, 32]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    # if num_classes == 2:
    #     activation = "sigmoid"
    #     units = 1
    # else:
    #     activation = "softmax"
    #     units = num_classes

    x = layers.Dropout(0.5)(x)
    # outputs = layers.Dense(units, activation=activation)(x)
    outputs = layers.Dense(num_classes)(x)
    return keras.Model(inputs, outputs)

def make_basic_model(input_shape, num_classes):
    inputs=keras.Input(shape=input_shape)
    # x = data_augmentation(inputs)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x =layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
    x =layers.MaxPooling2D(pool_size=(2, 2))(x)
    x =layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x =layers.MaxPooling2D(pool_size=(2, 2))(x)
    x =layers.Flatten()(x)
    x =layers.Dropout(0.5)(x)
    outputs =layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)



#%% Dataset generator
imgshape=(512,512)
batch_size=32
trainpath='/home/arun/Documents/PyWSPrecision/datasets/cat_surveytek_1/train'
validpath='/home/arun/Documents/PyWSPrecision/datasets/cat_surveytek_1/valid'
outputpath='/home/arun/Documents/PyWSPrecision/Pyoutputs/surveytek_outputs/'
model_filename='VGG16_model_defect_classifier.h5'
dot_img_file = 'MNIST_Model_1.png'
modelimgfilepath=os.path.join(outputpath, dot_img_file)
opfilepath=os.path.join(outputpath, model_filename)


train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    trainpath,
    subset="training",
    validation_split=0.2,
    seed=1337,#transformation seed
    shuffle=True,
    interpolation='nearest',
    label_mode="categorical",
    image_size=imgshape,
    batch_size=batch_size)

validation_ds=tf.keras.preprocessing.image_dataset_from_directory(
    validpath,
    subset="validation",
    validation_split=0.2,
    seed=1337,#transformation seed
    shuffle=True,
    interpolation='nearest',
    label_mode="categorical",
    image_size=imgshape,
    batch_size=batch_size)
#%%
class_names=train_ds.class_names
num_classes=len(class_names)

#%% Configuring dataset for performance
train_ds = train_ds.prefetch(buffer_size=32)
validation_ds = validation_ds.prefetch(buffer_size=32)
# os.system("nvidia-smi")
# print('After Dataset generator')
#%% Data Augumentation - Built in the future
"""tf.keras.preprocessing.image.ImageDataGenerator(
   rotation_range=30, horizontal_flip=True) Does not suit here"""
# data_augmentation = keras.Sequential(
#     [
#         layers.experimental.preprocessing.RandomFlip(mode="HORIZONTAL_AND_VERTICAL"),
#         layers.experimental.preprocessing.RandomRotation(0.1),
#     ]
# )

#%% Build models
input_shape=imgshape + (3,)
model = make_basic_model(input_shape, num_classes)
# model = make_simple_model(input_shape, num_classes)
# model = make_VGG16_model(input_shape, num_classes)
# os.system("nvidia-smi")
print('After Model-Built')
model.summary()
#%% Training the models
epochs = 50
flag=True
# callbacks = [
#     keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
# ]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
# os.system("nvidia-smi")
# print('Before Training')

# history=model.fit(
#     train_ds, epochs=epochs, validation_data=validation_ds,
# )
# if flag==True:
#     model.save(opfilepath)


tf.keras.utils.plot_model(model, to_file=modelimgfilepath, show_shapes=True)

    
# os.system("nvidia-smi")
# print('After Training')
#%%
# from matplotlib import pyplot as plt
# plt.figure(1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# plt.figure(2)
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#%%
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)
f.close()