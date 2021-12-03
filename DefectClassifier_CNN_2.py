#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 09:30:01 2021

@author: arun
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
import os
import sys
import PIL
from PIL import Image
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# print(device_lib.list_local_devices())

# cfg = tf.compat.v1.ConfigProto() 
# cfg.gpu_options.allow_growth = True
# sess= tf.compat.v1.Session(config=cfg)
#%%
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
f=open('console_output.txt','w+')
os.system("nvidia-smi")
print('Init')
#%% Dataset generator
imgshape=(256,256)
batch_size=32
trainpath='/home/arun/Documents/PyWSPrecision/datasets/cat_surveytek_1/train'
validpath='/home/arun/Documents/PyWSPrecision/datasets/cat_surveytek_1/valid'

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        # rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


train_ds=datagen.flow_from_directory(
    trainpath,
    # subset="training",
    seed=1337,#transformation seed
    shuffle=True,
    interpolation='nearest',
    class_mode="categorical",
    target_size=imgshape,
    batch_size=batch_size)

validation_ds=datagen.flow_from_directory(
    validpath,
    # subset="validation",
    seed=1337,#transformation seed
    shuffle=True,
    interpolation='nearest',
    class_mode="categorical",
    target_size=imgshape,
    batch_size=batch_size)
#%%
# class_names=train_ds.class_names
# num_classes=len(class_names)
num_classes=4 # Need to find the no. of class from the object
#%% Configuring dataset for performance
# train_ds = train_ds.prefetch(buffer_size=32)
# validation_ds = validation_ds.prefetch(buffer_size=32)
# os.system("nvidia-smi")
# print('After Dataset generator')
#%% Data Augumentation - Built in the future
"""tf.keras.preprocessing.image.ImageDataGenerator(
   rotation_range=30, horizontal_flip=True)"""
#%% CNN Models

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
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

def make_basic_model(input_shape, num_classes):
    inputs=keras.Input(shape=input_shape)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x =layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
    x =layers.MaxPooling2D(pool_size=(2, 2))(x)
    x =layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x =layers.MaxPooling2D(pool_size=(2, 2))(x)
    x =layers.Flatten()(x)
    x =layers.Dropout(0.5)(x)
    outputs =layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)
#%% Build models
input_shape=imgshape + (3,)
# model = make_basic_model(input_shape, num_classes)
model = make_simple_model(input_shape, num_classes)
# os.system("nvidia-smi")
print('After Model-Built')
model.summary()
#%% Training the models
epochs = 1

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

model.fit(
    train_ds, epochs=epochs, validation_data=validation_ds,
)
# os.system("nvidia-smi")
# print('After Training')
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