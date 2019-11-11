from PIL import Image
import os
import numpy as np
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pdb
import cv2
from mpl_toolkits.mplot3d import Axes3D

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

def get_data_bones(path):
    """
    Labels all data with bone for class assignment
    for unsupervised learning on all bones.
    """
    all_images_as_array=[]
    label=[]
    # pdb.set_trace()
    for filename in os.listdir(path):
        try:
            if re.match(r'ELBOW',filename):
                label.append(0)
            elif re.match(r'FINGER',filename):
                label.append(1)
            elif re.match(r'FOREARM',filename):
                label.append(2)
            elif re.match(r'HAND',filename):
                label.append(3)
            elif re.match(r'HUMERUS',filename):
                label.append(4)
            elif re.match(r'SHOULDER',filename):
                label.append(5)
            elif re.match(r'WRIST',filename):
                label.append(6)
            img=cv2.imread(path + filename)
            (b, g, r)=cv2.split(img)
            img=cv2.merge([r,g,b])
            np_array = np.asarray(img)
            l,b,c = np_array.shape
            np_array = np_array.reshape(l*b*c,)
            all_images_as_array.append(np_array)
        except:
            continue
    return np.array(all_images_as_array), np.array(label)

# Utility function to visualize the outputs of PCA and t-SNE

def scatter_pca(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)], legend='full')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []
    # pdb.set_trace()
    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

if __name__ == "__main__":
    pass