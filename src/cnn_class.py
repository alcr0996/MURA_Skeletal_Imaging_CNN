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


class binary_CNN(object):

    def __init__(self, train_path, val_path, test_path, model_name=None, new_model=True, previous_weights=None):

        self.weight_path = weight_path
        self.model_name = model_name
        self.model_save_path = "data/cnn_models/"
        self.metrics_save_path = "data/cnn_metrics/"
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.model_type = model_type

    def model_init(self):
        self.param_init()
        self.create_generators()
        self.make_callbacks()
   
    def fit(self):
        self.hist = self.model.fit_generator(
            self.train_gen,
            steps_per_epoch=self.n_train/self.batch,
            epochs=self.epochs,
            verbose=1,
            validation_data=self.val_gen,
            validation_steps=self.n_val/self.batch,
            use_multiprocessing=True,
            callbacks = self.callbacks
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
    
        # start back up here!!!! - maybe add final layer?
        #     if self.weight_path is not None:
        #     self.model.load_weights(self.weight_path, by_name=True, skip_mismatch=True)
        # self.model.compile( - going to have to def this
        #     loss='categorical_crossentropy', 
        #     optimizer='adam', 
        #     metrics=['accuracy', 
        #     self.top_3_accuracy, 
        #     self.top_5_accuracy])

    def create_generators(self, augmentation_strength=0.1, rotation_range=20, rescale=1./255, class_mode='binary', shuffle=False):
        
        train_datagen = ImageDataGenerator(
            rotation_range=rotation_range,
            rescale=rescale,
            width_shift_range=augmentation_strength,
            height_shift_range=augmentation_strength,
            shear_range=augmentation_strength,
            zoom_range=augmentation_strength*2,
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
