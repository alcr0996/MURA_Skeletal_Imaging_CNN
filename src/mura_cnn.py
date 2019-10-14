import os
os.environ['TF_KERAS']='1'

import numpy as np
np.random.seed(1337)  # for reproducibility

import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from keras_radam import RAdam
# train_op = RAdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999, weight_decay=0.0)

if __name__ == "__main__":
    
    # df = pd.read_csv('data/train_dir_paths/')
    # y = df.pop('target')
    # X = df
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # dimensions of our images.
    img_width, img_height = 150, 150

    train_data_dir = '../MURA_images/train/XR_FOREARM'
    validation_data_dir = '../MURA_images/valid/XR_FOREARM'
    nb_train_samples = 2000
    nb_validation_samples = 800
    epochs = 50
    batch_size = 16

    model = Sequential()
    model.add(Conv2D(32, (1, 1), input_shape=(2,)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
              metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


    validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

    model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

    model.save_weights('first_try.h5')

    validation_generator.class_indices     



    

