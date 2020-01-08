# To make RAdam work
import os
os.environ['TF_KERAS']='1'
# RAdam
from keras_radam import RAdam

import numpy as np
np.random.seed(42)  # for reproducibility
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import PReLU, LeakyReLU
from tensorflow.keras.callback import ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall

from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split


class CNN_creator(object):

    def __init__(self, train_path, val_path, model_name=None, new_model=True, previous_weights=None):

        self.weight_path = weight_path
        self.model_name = model_name
        self.model_save_path = "data/cnn_models/"
        self.metrics_save_path = "data/cnn_metrics/"
        self.train_path = train_path
        self.val_path = val_path
        self.model_type = model_type

    def model_init(self):
        self.param_init()
        self.create_generators()
   
    def fit(self):
        self.hist = self.model.fit_generator(
            self.train_gen,fourth_
            steps_per_epoch=self.n_train/self.batch,
            epochs=self.epochs,
            verbose=1,
            validation_data=self.val_gen,
            validation_steps=self.n_val/self.batch,
            use_multiprocessing=True,
            )

    def add_cnn_layer(self, num_filters = 32, kernel_size=(3, 3), pool_size=(2, 2), dropout=0.1,
                         num_blocks=1, first_layer=False, activation='relu', padding = 'same',
                         custom_weights=None, layer_number=None):
        
        if self.first_layer == True:
            self.model = Sequential() 
            self.model.add(Conv2D(num_filters, (kernel_size[0], kernel_size[1]), padding=padding,
                             name=layer_number+'_cnn_layer'))
            self.model.add(Activation(activation, name=layer_number+'_cnn_activaiton_layer'))
        else:
            self.model.add(Conv2D(num_filters, (kernel_size[0], kernel_size[1]), padding=padding,
                             name=layer_number+'_cnn_layer'))
            self.model.add(Activation(activation, name=layer_number+'_cnn_activaiton_layer'))

    def add_pooling_layer(self,pool_size=(2, 2), layer_name = None):
        self.model.add(MaxPooling2D(pool_size=(2, 2), name = layer_name+'pooling_layer')
    
    def add_dense_layer(self, num_neurons = 32, kernel_size=(3, 3), pool_size=(2, 2), dropout=0.1,
                         num_blocks=1, first_layer=False, activation=LeakyRelu(), padding = 'same',
                         custom_weights=None, layer_name=None):

        if self.first_layer == True:
            self.model.add(Flatten()) 
            self.model.add(Dense(num_neurons, (kernel_size[0], kernel_size[1]), padding=padding,
                             name=layer_number+'_dense_layer'))
            self.model.add(Activation(activation, name=layer_number+'_dense_activation_layer'))
        else:
            self.model.add(Dense(num_neurons, (kernel_size[0], kernel_size[1]), padding=padding,
                             name=layer_number+'_dense_layer'))
            self.model.add(Activation(activation, name=layer_number+'_dense_activation_layer'))
    
    def add_final_layer(self, num_neurons = 1, activation='sigmoid', optimizer='adam',
                         loss = 'binary_crossentropy', metrics=['accuracy'],
                         layer_name =None):
        self.model.add(Dense(num_neurons, name = layer_name+'final_dense_layer))
        self.model.add(Activation(activation, name=layer_name+'final_activation_layer'))

        self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)

    def save_model(self):
            model_path = self.model_save_path + self.model_name + ".h5"
            self.model.save(model_path)
            print("Saved model to \"" + model_path + "\"")

    def create_generators(self, augmentation_strength=0.1, rotation_range=20, rescale=1./255, class_mode='binary', shuffle=False):
        
        train_datagen = ImageDataGenerator(
            rotation_range=rotation_range,
            rescale=rescale,
            width_shift_range=augmentation_strength,
            height_shift_range=augmentation_strength,
            shear_range=augmentation_strength,
            zoom_range=augmentation_strength*2,num_neurons = 32,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.3, 0.7],
            )
        
        self.train_gen = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.image_size,
            batch_size=self.batch,
            class_mode=class_mode,
            shuffle=True)

        test_datagen = ImageDataGenerator(
            rescale=rescale
            )
        
        self.test_gen = test_datagen.flow_from_directory(
            self.test_path,
            target_size=self.image_size,
            batch_size=self.batch,
            class_mode=class_mode,
            shuffle=shuffle
            )

        self.val_gen = test_datagen.flow_from_directory(
            self.val_path,
            target_size=self.image_size,
            batch_size=self.batch,
            class_mode=class_mode,
            shuffle=shuffle
            )

if __name__ == "__main__":
    
    train_path = "../data/train_images/ELBOW"
    val_path = "../data/valid_images/ELBOW"

    model = CNN_creator(train_path, val_path, model_name='sigmoid_cnn')

    model.param_init(epochs=30, batch_size=20, image_size=(96, 96))
