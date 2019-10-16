# To make RAdam work
import os
os.environ['TF_KERAS']='1'
# RAdam
from keras_radam import RAdam

import numpy as np
np.random.seed(1337)  # for reproducibility

import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import PReLU, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import plot_model

from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    
    # df = pd.read_csv('data/train_dir_paths/')
    # y = df.pop('target')
    # X = df
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # dimensions of our images.
    img_width, img_height = 64, 64
    train_data_dir = 'data/train_images/all_train'
    validation_data_dir = 'data/valid_images/all_valid'
    nb_train_samples = 20000
    nb_validation_samples = 10000
    epochs = 10
    batch_size = 500

    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), input_shape=(img_width, img_height, 3),padding='valid', name = 'first_cnn_layer'))
    model.add(Activation('relu', name = 'first_cnn_activation'))
    model.add(MaxPooling2D(pool_size=(2, 2), name = 'first_pooling_layer'))

    model.add(Conv2D(128, (3, 3), padding='same', name = 'second_cnn_layer'))
    model.add(Activation('relu', name = 'second_cnn_activation'))
    model.add(MaxPooling2D(pool_size=(2, 2), name = 'second_pooling_layer'))

    model.add(Conv2D(256, (3, 3), padding='same', name = 'third_cnn_layer'))
    model.add(Activation('relu', name = 'third_cnn_activation'))
    model.add(MaxPooling2D(pool_size=(2, 2), name = 'third_pooling_layer'))

    model.add(Conv2D(256, (3, 3), padding='same', name = 'fourth_cnn_layer'))
    model.add(Activation('relu', name = 'fourth_cnn_activation'))
    model.add(MaxPooling2D(pool_size=(2, 2), name = 'fourth_pooling_layer'))

    # model.add(Conv2D(512, (3, 3), padding='same', name = 'fifth_cnn_layer'))
    # model.add(Activation('relu', name = 'fifth_cnn_activation'))
    # model.add(MaxPooling2D(pool_size=(2, 2), name = 'fifth_pooling_layer'))

    model.add(Flatten())
    
    model.add(Dense(64, name = 'first_dense_layer'))
    model.add(Activation(LeakyReLU(), name = 'first_dense_activation'))
    model.add(Dropout(0.1, name = 'first_dense_dropout'))
    
    model.add(Dense(32, name = 'second_dense_layer'))
    model.add(Activation(LeakyReLU(), name = 'second_dense_activation'))
    model.add(Dropout(0.1, name = 'second_dense_dropout'))

    model.add(Dense(1, name = 'final_sigmoid_layer'))
    model.add(Activation('sigmoid', name = 'sigmoid_activation'))

    # opt = Adam(learning_rate = 0.0001)
    model.compile(loss='binary_crossentropy',optimizer='adam',
                metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    
    # model.load_weights('data/model_weights/sigmoid_cnn.h5')         
    model_name = 'all_classes_sigmoid_cnn_64_64'
    model.save_weights('data/model_weights/'+model_name+'.h5')
    model.save('data/cnn_models/'+model_name+'.h5')




    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
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

    history = model.fit_generator(
    train_generator,
    steps_per_epoch= nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

    validation_generator.class_indices
    
    # plot loss during training
    fig, ax = plt.subplots(2, figsize = (12, 8))
    ax[0].set_title('Loss')
    ax[0].set_xticks(range(0,100,10))
    ax[0].plot(history.history['loss'], label='train')
    ax[0].plot(history.history['val_loss'], label='test')
    ax[0].legend()
    
    # plot accuracy during training
    ax[1].set_xticks(range(0,100,10))
    ax[1].set_title('Accuracy')
    ax[1].plot(history.history['accuracy'], label='train')
    ax[1].plot(history.history['val_accuracy'], label='test')
    ax[1].legend()
    
    plt.savefig('sigmoid_all_classes_cnn_positive_negative_64_64_2.png')
    
    # Confusion Matrix and Classification Report
    Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
    y_pred = np.where(Y_pred>=.5, 1, 0)
    print('Confusion Matrix')
    cm = confusion_matrix(validation_generator.classes, y_pred)
    print (cm)

    fpr, tpr, thresholds = roc_curve(validation_generator.classes, Y_pred)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_title('ROC - All Classes Positive vs. Negative 64_64_2')
    plt.savefig('ROC_all_classes_sigmoid_pos_vs_neg_64_64_2.png')

    model.summary()
    # print('Classification Report')
    # target_names = ['positive', 'negative']
    # print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

    # plot_model(model, to_file=model_name+'model_plot.png', show_shapes=True, show_layer_names=True)