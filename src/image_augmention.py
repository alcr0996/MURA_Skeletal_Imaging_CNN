from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import color, transform, restoration, io, feature
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pandas as pd 
import glob

def generate_folder_counts(class_names, dm, folder):
    """
    Returns the file count for each style in a given folder (train, test, val) in the form of a dictionary
    where the keys are styles.

    INPUTS:
        class_names - list of classes -list
        dm - directory path - str
        folder - train, test, or val - str
    OUTPUTS:
        image_count_dict - file counts for each class - dict
    """
    
    image_count_dict = dict()
    
    for c in class_names:
        directory_name = dm + folder + '/' + c
        image_count_dict[c] = len([name for name in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name, name))])
    return image_count_dict

def balance_classes(datagen, class_names, directory_name='data/train_images/'):
    """ 
    Will loop through the train test val folders and will create augmented images for the minority class 
    until all classes are balanced. 
    """
    folders = ['ELBOW', 'FINGER', 'HAND', 'WRIST', 'FOREARM', 'HUMERUS', 'SHOULDER']
    
    for folder in folders:
        image_count_dict = generate_folder_counts(class_names, directory_name, folder)
        
        print("\n\n\n\n\n\n\n\n")
        print(folder)
        print(image_count_dict)
        
        #Loops until all folders are equal in number
        while min(image_count_dict.values()) != max(image_count_dict.values()):
            image_count_dict = generate_folder_counts(class_names, directory_name, folder)

            count = max(image_count_dict.values()) - min(image_count_dict.values())
            minority_class = min(image_count_dict, key=image_count_dict.get)
            
            max_count = max(image_count_dict.values())
            min_count = min(image_count_dict.values())
            
            print(minority_class)
            print(max_count)
            print(min_count)

            minority_file_path = directory_name + folder + '/' + minority_class +"/"
            minority_list = [name for name in os.listdir(minority_file_path) if os.path.isfile(os.path.join(minority_file_path, name))]
           
            for i in range(count):
                try:
                    image_path = minority_file_path + minority_list[i]
                    img = io.imread(image_path)  
                    x = img_to_array(img)  
                    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

                    #Generates one augmented image before moving onto the next one
                    for batch in datagen.flow(x, batch_size=1,save_to_dir=minority_file_path, save_prefix='altered', save_format='jpg'):
                        break
                except:
                  continue

def 
if __name__ == '__main__':
    
    datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    class_names = ['abstract', 'brush','bubble', 'cartoon','realistic', 'wildstyle']

    balance_classes(datagen, class_names, './data/train_test_split/')