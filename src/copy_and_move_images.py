import shutil
import os
import os.path
import pdb
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img, array_to_img, img_to_array, load_img
import random
from skimage import color, transform, restoration, io, feature



def copy_move_files(path, target_dir):
    word_list = ['ELBOW','FINGER', 'HAND', 'WRIST', 'FOREARM', 'HUMERUS', 'SHOULDER'] 
    for word in word_list:
        i = 1
        directory_list = [x[0] for x in os.walk(path+'XR_'+word)]
        for directory in directory_list:
            if 'positive' in directory:
                for root, direct, files in os.walk(directory):
                    for f in files:
                        shutil.copy(root+'/'+f, target_dir+word+'/positive/image'+str(i)+'.png')
                        if i == 21:
                            print(root)
                            print(direct)
                        i += 1

            elif 'negative' in directory:
                for root, direct, files in os.walk(directory):
                    for f in files:
                        # base, extension = os.path.splitext(filename)
                        shutil.copy(root+'/'+f, target_dir+word+'/negative/image'+str(i)+'.png')
                        if i == 21:
                            print(root)
                            print(direct)
                        i += 1
            else:
                continue

def copy_move_files_all_class(path, target_dir):
    i = 1
    directory_list = [x[0] for x in os.walk(path)]
    for directory in directory_list:
        if 'positive' in directory:
            for root, direct, files in os.walk(directory):
                for f in files:
                    shutil.copy(root+'/'+f, target_dir+'/positive/image'+str(i)+'.png')
                    i += 1
                    print(i)
        elif 'negative' in directory:
            for root, direct, files in os.walk(directory):
                for f in files:
                    # base, extension = os.path.splitext(filename)
                    shutil.copy(root+'/'+f, target_dir+'/negative/image'+str(i)+'.png')
                    i += 1
                    print(i)
        else:
            continue

def copy_move_files_all_bones(path, target_dir):
    word_list = ['ELBOW', 'FINGER', 'HAND', 'WRIST', 'FOREARM', 'HUMERUS', 'SHOULDER'] 
    for word in word_list:
        i = 1
        directory_list = [x[0] for x in os.walk(path)]
        for directory in directory_list:
            if word in directory:
                for root, direct, files in os.walk(directory):
                    for f in files:
                        shutil.copy(root+'/'+f, target_dir+'/'+word+'/image'+str(i)+'.png')
                        i += 1
                        print(i)
            else:
                continue

def balance_classes(datagen, directory, word_list): 
    for word in word_list: 
        difference = 0
        count_classes = [len(files) for r, d, files in os.walk(directory+'/'+word)]
        for i in range(len(count_classes)):
            if 0 in count_classes:
                count_classes.pop(0)
        difference = abs(count_classes[0] - count_classes[1])
        if count_classes[0] < count_classes[1]:
            for r, d, files in os.walk('data/train_images/'+word+'/negative'):
                for i in range(difference):
                    filepath = 'data/train_images/'+word+'/negative/'
                    img_name = random.choice(files)
                    img = io.imread(filepath+img_name)
                    x = img_to_array(img)
                    x = x.reshape((1,) + x.shape)
                    dumb = datagen.flow(x, batch_size=1,
                                save_to_dir=directory+'/'+word+'/negative/',
                                save_prefix='altered_negative', save_format='png'
                                )
                    dumb[0]
        elif count_classes[1] < count_classes[0]:
            for r, d, files in os.walk('data/train_images/'+word+'/positive'):
                for i in range(difference):
                    filepath = 'data/train_images/'+word+'/positive/'
                    img_name = random.choice(files)
                    img = io.imread(filepath+img_name)
                    x = img_to_array(img)
                    x = x.reshape((1,) + x.shape)
                    dumb = datagen.flow(x, batch_size=1,
                                save_to_dir=directory+'/'+word+'/positive/',
                                save_prefix='altered_positive_3', save_format='png'
                                )
                    dumb[0]


if __name__ == "__main__":
    # copy_move_files('MURA_images/train/', 'data/train_images/')
    # copy_move_files('MURA_images/valid/', 'data/valid_images/')

    copy_move_files_all_class('data/train_images', 'data/all_train')
    # copy_move_files_all_class('data/valid_images', 'data/all_valid')

    # copy_move_files_all_bones('MURA_images/train', 'data/train_images/all_bones_train')
    # copy_move_files_all_bones('MURA_images/valid', 'data/valid_images/all_bones_valid')

    # this is the augmentation configuration used for training
    datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
            )
    
    bone_list = ['ELBOW', 'FINGER', 'HAND', 'WRIST', 'FOREARM', 'HUMERUS', 'SHOULDER']
    # balance_classes(datagen, 'data/train_images', bone_list)
