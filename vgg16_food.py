#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tuesday Dec 04 18:15:13 2018

@author: lingweibao

"""

import keras
from keras.applications  import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import os
import numpy
from PIL import Image
from keras.utils import np_utils

train_dir = '/Users/lingweibao/Documents/Bao/05-Source/food_recognition/foods/train'
val_dir = '/Users/lingweibao/Documents/Bao/05-Source/food_recognition/foods/val'

batch_size = 20
WIDTH = 150
HEIGHT = 150
epochs = 7

convVGG16 = VGG16(weights='imagenet', include_top=False, input_shape=(WIDTH, HEIGHT, 3))
convVGG16.summary()

model = Sequential()
model.add(convVGG16)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))

convVGG16.trainable = False

train_idg = ImageDataGenerator(
      rescale=1.0/255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_idg = ImageDataGenerator(rescale=1.0/255)

train_generator = train_idg.flow_from_directory(
        train_dir,
        target_size=(WIDTH, HEIGHT),
        batch_size=batch_size)

val_generator = test_idg.flow_from_directory(
        val_dir,
        target_size=(WIDTH, HEIGHT),
        batch_size=batch_size)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=50,
      epochs=epochs,
      validation_data=val_generator,
      validation_steps=20,
      verbose=1)

model.save('food_vgg16_weight.h5')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
print('Test acc:', acc)
print('Test val_acc:', val_acc)
print('Test loss:', loss)
print('Test val_loss:', val_loss)