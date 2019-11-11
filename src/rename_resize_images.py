from PIL import Image
import os
import numpy as np
import pandas as pd
import re
import pdb


def resize_multiple_images(src_path, dst_path):
    """
    Resize images in a directory to (96, 96)
    src_path: directory images are in
    dst_path: directory to copy and move images to
    """
    for filename in os.listdir(src_path):
        img=Image.open(src_path+'/'+filename)
        new_img = img.resize((96,96))
        #new_img.resize(96,96,1)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        new_img.save(dst_path+'/'+filename)
        print('Resized and saved {} successfully.'.format(filename))



def rename_multiple_files(path,obj):
    """
    Rename image files with class.
    path: directory of images to rename
    obj: class name to add to beginning of image filename
    """
    i=0
    for filename in os.listdir(path):
        try:
            f,extension = os.path.splitext(path+filename)
            src=path+filename
            dst=path+obj+str(i)+extension
            os.rename(src,dst)
            i+=1
            print('Rename successful.')
        except:
            i+=1

if __name__ == "__main__":

    #resize
    src_path = "data/train_images/ELBOW/negative"
    dst_path = "data/train_images/PCA/ELBOW/negative"
    resize_multiple_images(src_path, dst_path)
    
    src_path = "data/train_images/ELBOW/positive"
    dst_path = "data/train_images/PCA/ELBOW/positive"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/train_images/FINGER/negative"
    dst_path = "data/train_images/PCA/FINGER/negative"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/train_images/FINGER/positive"
    dst_path = "data/train_images/PCA/FINGER/positive"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/train_images/FOREARM/negative"
    dst_path = "data/train_images/PCA/FOREARM/negative"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/train_images/FOREARM/positive"
    dst_path = "data/train_images/PCA/FOREARM/positive"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/train_images/HAND/negative"
    dst_path = "data/train_images/PCA/HAND/negative"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/train_images/HAND/positive"
    dst_path = "data/train_images/PCA/HAND/positive"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/train_images/HUMERUS/negative"
    dst_path = "data/train_images/PCA/HUMERUS/negative"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/train_images/HUMERUS/positive"
    dst_path = "data/train_images/PCA/HUMERUS/positive"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/train_images/SHOULDER/negative"
    dst_path = "data/train_images/PCA/SHOULDER/negative"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/train_images/SHOULDER/positive"
    dst_path = "data/train_images/PCA/SHOULDER/positive"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/train_images/WRIST/positive"
    dst_path = "data/train_images/PCA/WRIST/positive"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/train_images/WRIST/negative"
    dst_path = "data/train_images/PCA/WRIST/negative"
    resize_multiple_images(src_path, dst_path)

    #rename
    #negatives

    path="data/train_images/PCA/ELBOW/negative/"
    obj='ELBOWnegative'
    rename_multiple_files(path,obj)

    path="data/train_images/PCA/FINGER/negative/"
    obj='FINGERnegative'
    rename_multiple_files(path,obj)

    path="data/train_images/PCA/FOREARM/negative/"
    obj='FOREARMnegative'
    rename_multiple_files(path,obj)

    path="data/train_images/PCA/HAND/negative/"
    obj='HANDnegative'
    rename_multiple_files(path,obj)

    path="data/train_images/PCA/HUMERUS/negative/"
    obj='HUMERUSnegative'
    rename_multiple_files(path,obj)

    path="data/train_images/PCA/SHOULDER/negative/"
    obj='SHOULDERnegative'
    rename_multiple_files(path,obj)

    path="data/train_images/PCA/WRIST/negative/"
    obj='WRISTnegative'
    rename_multiple_files(path,obj)

    #rename
    #positives

    path="data/train_images/PCA/ELBOW/positive/"
    obj='ELBOWpositive'
    rename_multiple_files(path,obj)

    path="data/train_images/PCA/FINGER/positive/"
    obj='FINGERpositive'
    rename_multiple_files(path,obj)
    
    path="data/train_images/PCA/FOREARM/positive/"
    obj='FOREARMpositive'
    rename_multiple_files(path,obj)

    path="data/train_images/PCA/HAND/positive/"
    obj='HANDpositive'
    rename_multiple_files(path,obj)

    path="data/train_images/PCA/HUMERUS/positive/"
    obj='HUMERUSpositive'
    rename_multiple_files(path,obj)

    path="data/train_images/PCA/SHOULDER/positive/"
    obj='SHOULDERpositive'
    rename_multiple_files(path,obj)

    path="data/train_images/PCA/WRIST/positive/"
    obj='WRISTpositive'
    rename_multiple_files(path,obj)