#Xufan
#TODO:
# writing a run script
# adding visualizations that can help select models / hyperparameters 

import tensorflow as tf
from tensorflow import keras
from keras.layers import (Conv2D, MaxPool2D, Dropout, Flatten, 
    Dense, BatchNormalization, RandomFlip, RandomContrast, RandomRotation, RandomZoom,LeakyReLU)
from keras.optimizers import Adam
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import preprocessing
from model import CNN_Model
import sys
import argparse
import re
from datetime import datetime
from preprocessing import generate_data
from skimage.transform import resize
from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def acc(predicted_classes, testing_labels):
    a = 0
    for i in range(len(predicted_classes)):
        if predicted_classes[i]==testing_labels[i]:
            a+=1
    acc = a / len(predicted_classes)
    return acc

def main():
    """ Main function. """
    training_data, testing_data, training_labels, testing_labels = preprocessing.generate_data(testing_size = 0.2, image_height=256, image_width=256, path='/data/images') 

    model = CNN_Model()
    architecture = Sequential([
                Conv2D(16, (2,2), padding='same'),
                LeakyReLU(),
                BatchNormalization(),
                MaxPool2D(pool_size=(2,2)),
                Conv2D(16, (2,2), padding='same'),
                LeakyReLU(),
                BatchNormalization(),
                MaxPool2D(pool_size=(2,2)),
                Conv2D(32, (2,2), padding='same'),
                LeakyReLU(),
                BatchNormalization(),
                MaxPool2D(pool_size=(2,2)),
                Conv2D(32, (2,2), padding='same'),
                LeakyReLU(),
                BatchNormalization(),
                MaxPool2D(pool_size=(2,2)),
                Dropout(0.1), 
                Conv2D(64, (2,2), padding='same'),
                LeakyReLU(),
                BatchNormalization(),
                MaxPool2D(pool_size=(2,2)),
                Conv2D(64, (2,2), padding='same'),
                LeakyReLU(),
                BatchNormalization(),
                MaxPool2D(pool_size=(2,2)),
                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.1), 
                Dense(21, activation="softmax")
                ])

    checkpoint_path = "training_1/cp.ckpt" #TODO: change to desired path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.listdir(checkpoint_dir)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    model.build_model(architecture=architecture, optimizer=Adam, learning_rate=0.001)
    print('Start training')
    model.train(train_data=training_data,train_labels=training_labels,augment=False,epochs=30,batch_size=10,callbacks=cp_callback)
    
    checkpoint_path = "checkpoints/acc_60/cp.ckpt" #TODO: change to desired path
    model.load_state(checkpoint_path)
    test_loss, test_accuracy = model.test(test_data=testing_data,test_labels=testing_labels)
    print("done testing")

    predicted_classes = np.argmax(model.model.predict(testing_data,batch_size=15),axis=1)
    print('predicted_classes')
    print(predicted_classes)

    print('***************************************************')
    print('testing_labels')
    print(testing_labels)

    print('accuracy:', acc(predicted_classes, testing_labels))  


main()

#TODO: 
# hook up to model / functions
# format would be similar to :
# generating data:
# train_data, train_labels = get_data()
# test_data, test_labels = get_data()
# CNNmodel.build()
# CNNmodel.train(train_data,train_labels)
# CNNmodel.test(test_data,test_labels) to get metrics