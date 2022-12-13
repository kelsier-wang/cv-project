import tensorflow as tf
from tensorflow import keras
from keras.layers import (Conv2D, MaxPool2D, Dropout, Flatten, 
    Dense, BatchNormalization, RandomFlip, RandomContrast, RandomRotation, RandomZoom, LeakyReLU)
from keras.optimizers import Adam
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
import preprocessing
import numpy as np

#Julianne
#TODO:
# construct cnn model(s)
# add augmentations, optimizer (Adam), loss / accuracy
# consider saving weights / being able to test from saved checkpoints, consider outputting confidence in match


class CNN_Model(tf.keras.Model):

    def __init__(self):
        super(CNN_Model, self).__init__()

        self.loss = SparseCategoricalCrossentropy

        #TODO: figure out how many output classes there are (possibly) units of model to accomodate
        self.num_classes = 21
        self.model = Sequential([])

        #TODO: possibly change / tune
        self.augment_fn = Sequential([
            RandomFlip("horizontal"),
            RandomFlip("vertical"),
            RandomRotation(0.2),
            RandomZoom(0.1)
        ])

    #TODO: test / modify  
    def build_model(self, architecture=None, optimizer=Adam, learning_rate=0.001):
        if architecture == None:
            self.model = Sequential([
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
                Dense(self.num_classes, activation="softmax")
            ])
        else:
            self.model = architecture
            
        self.model.compile(
            optimizer=optimizer(learning_rate),
            loss = self.loss_fn,
            metrics='sparse_categorical_accuracy'
        )


    def loss_fn(self, labels, predictions):
        return self.loss()(labels,predictions)

    #TODO: possibly change default params
    def train(self, train_data, train_labels, augment=True, epochs=10, batch_size=10, callbacks = None):

        if augment:
            train_data = self.augment_fn(train_data)

        training_metrics = self.model.fit(x=train_data,y=train_labels,epochs=epochs,batch_size=batch_size, callbacks=callbacks)
        return training_metrics

    def test(self, test_data, test_labels):
        evaluation_metrics = self.model.evaluate(x=test_data,y=test_labels)
        return evaluation_metrics

    def save_state(self, path):
        self.model.save_weights(path)
    
    def load_state(self, path):
        self.model.load_weights(path)
