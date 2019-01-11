# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.preprocessing import ImageDataGenerator
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
"label_bin": "/informatik2/students/home/4banik/PycharmProjects/Bird_Detection/out/plot.png"
}

#grab the image paths and randomly shuffle
trainImagePaths = np.genfromtxt(args["metaDir"]+"train.txt", dtype=None)
testImagePaths = np.genfromtxt(args["meta"]+"test.txt", dtype=None)
random.seed(42)
random.shuffle(trainImagePaths)
random.shuffle(testImagePaths)

#load images
#data pre-processing
#train images
trainData = []
trainLabels = []
for imagePath in trainImagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64,64))
    trainData.append(image)
    label = imagePath.split(os.path.sep)[-2]
    trainLabels.append(label)

#val and test images
valData = []
testData = []
valLabels = []
testLabels = []
idx = 0
testSize = len(testImagePaths)
for imagePath in testImagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (imageSize,imageSize))
    label = imagePath.split(os.path.sep)[-2]
    if idx < testSize/2:
        valData.append(image)
        valLabels.append(label)
    else:
        testData.append(image)
        testLabels.append(label)
    idx = idx + 1

#normalize data to the range[0,1]
trainX = np.array(trainData, dtype="float")/255.0
trainY = np.array(trainLabels)
valX = np.array(valData, dtype="float")/255.0
valY = np.array(valLabels)
testX = np.array(testData, dtype="float")/255.0
testY = np.array(testLabels)

#convert label to vector form
classes = np.unique(trainY)
trainY, valY, testY = (label_binarize(Y, classes) for Y in (trainY, valY, testY))

#data augmentation: construct image generator
trainDataAug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True,
    fill_mode="nearest")

#load model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))
#freeze layer
for layer in vgg_conv.layers[-4]:
    layer.trainable = False

for layer in vgg_conv.layers:
    print(layer, layer.trainable)

model = Sequential()
model.add(vgg_conv)
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax')
model.summary()

# initialize our initial learning rate, # of epochs to train for,
# and batch size
INIT_LR = 0.01
EPOCHS = 75
BS = 32

# initialize the model and optimizer
opt = SGD(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(optimizer=opt, loss="categorical_crossentropy",
    metrics=["accuracy"])

# train network
H = model.fit_generator(trainDataAug.flow(trainX, trainY, batch_size=BS),
    validation_data=(valX, valY),
    steps_per_epoch = len(trainY)//BS,
    epochs= EPOCHS,
    use_multiprocessing=True,
    verbose=2)

# evaluate network
print("[INFO] evaluating network")
predictions = model.predict(valX, batch_size=32)
print(classification_report(valY.argmax(axis=1), predictions.argmax(axis=1), target_names=classes))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()

