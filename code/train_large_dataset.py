# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

args = {
"dataset": "/informatik2/students/home/4banik/Documents/datasets/CUB_200/images/",
"model": "/data_b/soubarna/models/bird_detection/bird_vgg16.model",
"plot": "/informatik2/students/home/4banik/PycharmProjects/Bird_Detection/out/plot.png",
"metaDir": "/informatik2/students/home/4banik/Documents/datasets/CUB_200/lists/",
"imageSize": 224,
"label_bin": "/informatik2/students/home/4banik/PycharmProjects/Bird_Detection/out/plot.png",
"train_dir":"/informatik2/students/home/4banik/Documents/datasets/CUB_200/train/",

}


# NETWORK DEFINITION
#load model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(args["imageSize"], args["imageSize"], 3))
#freeze layer
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

for layer in vgg_conv.layers:
    print(layer, layer.trainable)

model = Sequential()
model.add(vgg_conv)
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))
model.summary()


#data augmentation: construct image generator
trainDataAug = ImageDataGenerator(rescale=1.0/255, rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True,
    fill_mode="nearest")

train_generator = trainDataAug.flow_from_directory(
        args["train_dir"],
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)
