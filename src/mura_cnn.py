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
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from keras_radam import RAdam
opt = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)

if __name__ == "__main__":
    
    # df = pd.read_csv('data/train_dir_paths/')
    # y = df.pop('target')
    # X = df
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # dimensions of our images.
    img_width, img_height = 200, 200
    train_data_dir = 'data/train_images/FOREARM'
    validation_data_dir = 'data/valid_images/FOREARM'
    nb_train_samples = 1830
    nb_validation_samples = 301
    epochs = 100
    batch_size = 50

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(img_width, img_height, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    # model.load_weights('fifth_large_try.h5')         

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.25,
            height_shift_range=0.25,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.25,
            horizontal_flip=True,
            fill_mode='nearest'
            )
    
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
    steps_per_epoch= nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

    model.save_weights('7th_large_try.h5')


    validation_generator.class_indices
    

    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show()
    
    # Confusion Matrix and Classification Report
    Y_pred = model.predict_generator(validation_generator, 301 // batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('Classification Report')
    target_names = ['positive', 'negative']
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


    

