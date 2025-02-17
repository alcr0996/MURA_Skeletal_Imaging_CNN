import os
from glob import glob

import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import precision_score , recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.metrics import Precision, Recall 


from build_transfer_model import create_transfer_model
from simple_cnn import create_model
from plotter import plot_confusion_matrix

# config = tensorflow.keras.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tensorflow.Session(config=config)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# session = tf.Session(config=config)


class ClassificationNet(object):
    """Keras Image Classifier with added methods to create directory datagens and evaluate on validation set
        """

    def __init__(self,  project_name, target_size, augmentation_strength=0.1, preprocessing=None, batch_size=16):
        """
        Initialize class with basic attributes

        Args:
        project_name (str): project name, used for saving models
        target_size (tuple(int, int)): size of images for input
        augmentation_strength (float): strength for image augmentation transforms
        batch_size(int): number of samples propogated throught network
        preprocessing(function(img)): image preprocessing function

            """
        self.project_name = 'Xception_class_170'
        self.target_size = target_size
        self.input_size = self.target_size + (3,) # target size with color channels
        self.train_datagen = ImageDataGenerator()
        self.validation_datagen = ImageDataGenerator()
        self.augmentation_strength = augmentation_strength
        self.train_generator = None
        self.validation_generator = None
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.class_names =  None

    def _init_data(self, train_folder, validation_folder):
        """
        Initializes class data

        Args:
            train_folder(str): folder containing train data
            validation_folder(str): folder containing validation data
            """
        self.train_folder = train_folder
        self.validation_folder = validation_folder

        self.nTrain = sum(len(files) for _, _, files in os.walk(self.train_folder)) #: number of training samples
        self.nVal = sum(len(files) for _, _, files in os.walk(self.validation_folder)) #: number of validation samples
        self.n_categories = sum(len(dirnames) for _, dirnames, _ in os.walk(self.train_folder)) #: number of categories
        self.class_names = self.set_class_names() #: text representation of classes

    def _create_generators(self):
        """
        Create generators to read images from directory
            """

        # Set parameters for processing and augmenting images
        self.train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocessing,
            rotation_range=15*self.augmentation_strength,
            width_shift_range=self.augmentation_strength,
            height_shift_range=self.augmentation_strength,
            shear_range=self.augmentation_strength,
            zoom_range=self.augmentation_strength
        )
        # no need for augmentation on validation images
        self.validation_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocessing
        )

        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_folder,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)

        self.validation_generator = self.validation_datagen.flow_from_directory(
            self.validation_folder,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False)

    def fit(self, train_folder, validation_folder, model_fxn, optimizer, epochs):
        """
        Fits the CNN to the data, then saves and predicts on best model

        Args:
            train_folder(str): folder containing train data
            validation_folder(str): folder containing validation data
            model_fxn(function): function that returns keras Sequential classifier
            optimizer(keras optimizer): optimizer for training
            epochs(int): number of times to pass over data

        Returns:
            str: file path for best modelvali
            """

        self._init_data(train_folder, validation_folder)
        print(self.class_names)
        self._create_generators()
        model = model_fxn(self.input_size, self.n_categories)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

        # Initialize tensorboard for monitoring
        tensorboard = tensorflow.keras.callbacks.TensorBoard(
            log_dir=self.project_name, histogram_freq=0, batch_size=self.batch_size, write_graph=True, embeddings_freq=0)
        if not os.path.exists('models'):
            os.makedirs('models')

        # Initialize model checkpoint to save best model
        savename = 'models/'+self.project_name+'.hdf5'
        mc = keras.callbacks.ModelCheckpoint(savename, monitor='val_loss', 
                                             verbose=0, save_best_only=True, 
                                             save_weights_only=False, mode='auto',
                                             period=1)

        history = model.fit_generator(self.train_generator,
                                      steps_per_epoch=self.nTrain/self.batch_size,
                                      epochs=epochs,
                                      validation_data=self.validation_generator,
                                      validation_steps=self.nVal/self.batch_size,
                                      callbacks=[mc, tensorboard])

        best_model = load_model(savename)
        print('evaluating simple model')
        accuracy = self.evaluate_model(best_model, self.validation_folder)
        return savename

    def evaluate_model(self, model, validation_folder):
        """
        evaluates model on validation data
        Args:
            model (keras classifier model): model to evaluate
            validation_folder (str): path of validation data
        Returns:
            list(float): metrics returned by the model, typically [loss, accuracy]
            """

        validation_generator = self.validation_datagen.flow_from_directory(
            validation_folder,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False)

        metrics = model.evaluate_generator(validation_generator,
                                           steps=self.nVal/self.batch_size,
                                           use_multiprocessing=True,
                                           verbose=1)
        print(f"validation loss: {metrics[0]} accuracy: {metrics[1]}")

        return metrics, validation_generator

    def print_model_layers(self, model, indices=0):
        """
        prints model layers and whether or not they are trainable

        Args:
            model (keras classifier model): model to describe
            indices(int): layer indices to print from
        Returns:
            None
            """

        for i, layer in enumerate(model.layers[indices:]):
            print(f"Layer {i+indices} | Name: {layer.name} | Trainable: {layer.trainable}")

    def process_img(self,img_path):
        """
        Loads image from filename, preprocesses it and expands the dimensions because the model predict function expects a batch of images, not one image
        Args:
            img_path (str): file to load
        Returns:
            np.array: preprocessed image
        """
        original = load_img(img_path, target_size = self.target_size)
        numpy_image = self.preprocessing(img_to_array(original))
        image_batch = np.expand_dims(numpy_image, axis =0)
        return image_batch

    def model_predict(self, img_path,model):
        """
        Uses an image and a model to return the names and the predictions of the top 3 classes

        Args:
            img_path (str): file to load
            model (keras classifier model): model to use for prediction

        Returns:
            str: top 3 predictions
            """
        im =  self.process_img(img_path)
        preds =  model.predict(im)
        top_3 = preds.argsort()[0][::-1][:3] # sort in reverse order and return top 3 indices
        top_3_names = self.class_names[top_3]
        top_3_percent = preds[0][[top_3]]*100
        top_3_text = '\n'.join([f'{name}: {percent:.2f}%' for name, percent in zip(top_3_names,top_3_percent)])
        return top_3_text

    def set_class_names(self):
        """
        Sets the class names, sorted by alphabetical order
        """
        names = [os.path.basename(x) for x in glob(self.train_folder + '/*')]
        return sorted(names)


class TransferClassificationNet(ClassificationNet):
    """Image Classifier Implementing Transfer Methods"""

    def fit(self, train_folder, validation_folder, model_fxn, optimizers, epochs, freeze_indices, warmup_epochs=5):
        """
        Fits the CNN to the data, then saves and predicts on best model

        Args:
            train_folder(str): folder containing train data
            validation_folder(str): folder containing validation data
            model_fxn(function): function that returns keras Sequential classifier
            optimizers(list(keras optimizer)): optimizers for training, first value is for warmup, second value is for training
            epochs(int): number of times to pass over data
            freeze_indices(list(int)): layer indices to freeze up to, first value is for warmup, second value is for training
            warmup_epochs(int): number of epochs to warm up head for

        Returns:
            str: file path for best model
            """
        self._init_data(train_folder, validation_folder)
        self._create_generators()

        model = model_fxn(self.input_size, self.n_categories)
        self.change_trainable_layers(model, freeze_indices[0])

        model.compile(optimizer=optimizers[0],
                      loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

        # Initialize tensorboard for monitoring
        tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir=self.project_name, 
                                                  histogram_freq=0, 
                                                  batch_size=self.batch_size, 
                                                  write_graph=True, 
                                                  embeddings_freq=0)

        if not os.path.exists('models'):
            os.makedirs('models')

        # Initialize model checkpoint to save best model
        savename = 'models/'+self.project_name+'.hdf5'
        mc = keras.callbacks.ModelCheckpoint(savename, monitor='val_loss', 
                                             verbose=0, save_best_only=True, 
                                             save_weights_only=False, mode='auto',
                                             period=1)

        history = model.fit_generator(self.train_generator,
                                      steps_per_epoch=self.nTrain/self.batch_size,
                                      epochs=warmup_epochs,
                                      validation_data=self.validation_generator,
                                      validation_steps=self.nVal/self.batch_size,
                                      callbacks=[mc, tensorboard])

        self.change_trainable_layers(model, freeze_indices[1])
        model.compile(optimizer=optimizers[1], loss='binary_crossentropy',
                      metrics=['accuracy', Precision(), Recall()])
        history = model.fit_generator(self.train_generator,
                                      steps_per_epoch=self.nTrain/self.batch_size,
                                      epochs=epochs,
                                      validation_data=self.validation_generator,
                                      validation_steps=self.nVal/self.batch_size,
                                      callbacks=[mc, tensorboard])
        best_model = load_model(savename)
        print('evaluating simple model')
        accuracy = self.evaluate_model(best_model, self.validation_folder)
        
        return savename

    def change_trainable_layers(self, model, trainable_index):
        """
        unfreezes model layers after passed index, freezes all before

        Args:
        model (keras Sequential model): model to change layers
        trainable_index(int): layer to split frozen /  unfrozen at

        Returns:
            None
            """

        for layer in model.layers[:trainable_index]:
            layer.trainable = False
        for layer in model.layers[trainable_index:]:
            layer.trainable = True


def main():
    train_folder = 'data/all_train'
    validation_folder = 'data/all_valid'
    # holdout_folder = 'food_data/data/holdout_small'

    target_size = (71, 71)  # 299,299 is suggested for xception but is quite taxing on cpu
    epochs = 10
    batch_size = 20

    model_fxn = create_model
    opt = Adam(lr=0.001)

    simple_cnn = ClassificationNet('simple_class_test', target_size, augmentation_strength=0.2,
                                   preprocessing=preprocess_input, batch_size=batch_size)
    
    simple_cnn.fit(train_folder, validation_folder, model_fxn, opt, epochs)
    model_fxn = create_transfer_model
    freeze_indices = [132, 126] # first unfreezing only head, then conv block 14
    optimizers = [Adam(lr=0.0006), Adam(lr=0.0001)] # keep learning rates low to keep from wrecking weights

    warmup_epochs = 0
    epochs = epochs - warmup_epochs
    transfer_model = TransferClassificationNet('transfer_test', target_size, 
                                                augmentation_strength=0.2, 
                                                preprocessing=preprocess_input, batch_size=batch_size)
    
    transfer_model.fit(train_folder, validation_folder, model_fxn,
                       optimizers, epochs, freeze_indices, warmup_epochs=warmup_epochs) 
                             
   


if __name__ == '__main__':
    main()