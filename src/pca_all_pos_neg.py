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

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

def resize_multiple_images(src_path, dst_path):
    """
    Resize images in a directory to (96, 96)
    src_path: directory images are in
    dst_path: directory to copy and move images to
    """
    for filename in os.listdir(src_path):
        img=Image.open(src_path+'/'+filename)
        new_img = img.resize((96,96,))
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

def get_data(path):
    """
    Labels all data with pos/neg for class assignment
    for unsupervised learning
    """
    all_images_as_array=[]
    label=[]
    for filename in os.listdir(path):
        try:
            if re.match(r'positive',filename):
                label.append(1)
            else:
                label.append(0)
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
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)], legend = 'full')
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

    #resize

    src_path = "data/all_train/negative"
    dst_path = "data/train_images/PCA/train/negative"
    resize_multiple_images(src_path, dst_path)

    src_path = "data/all_train/positive"
    dst_path = "data/train_images/PCA/train/positive"
    resize_multiple_images(src_path, dst_path)

    #rename

    path="data/train_images/PCA/train/negative/"
    obj='negative'
    rename_multiple_files(path,obj)

    path="data/train_images/PCA/train/positive/"
    obj='positive'
    rename_multiple_files(path,obj)

    # # create training data
    train_data = "data/train_images/PCA/all_train/images/"
    X_train, y_train = get_data(train_data)

    time_start = time.time()

    x_subset = X_train[0:20000]
    y_subset = y_train[0:20000]

    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(X_train)

    print ('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))

    pca_df = pd.DataFrame(columns = ['pca1','pca2'])

    pca_df['pca1'] = pca_result[:,0]
    pca_df['pca2'] = pca_result[:,1]

    print ('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))
    print(f'Total variance explained by 50 components: {sum(pca.explained_variance_ratio_)}')
    
    # taking top two principal components
    top_two_comp = pca_df[['pca1','pca2']]
    
    # Visualizing the PCA output
    scatter_pca(top_two_comp.values, y_train)
    plt.savefig('pca_2_comp_all_train.png')

    # T-SNE on all datapoints - takes forever/don't run 
    # without ample time (hours) unless aggressively subsetting data
    tsne_all = TSNE(random_state=RS, verbose=1).fit_transform(x_subset)
    scatter_pca(tsne_all, y_subset)
    plt.savefig('tsne_all_train_subset.png')
    
    # T-SNE w/ PCA 
    time_start = time.time()
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(X_train)
    print ('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
    pca_tsne = TSNE(random_state=RS, verbose=1).fit_transform(pca_result_50)
    
    # Visualize T-SNE w/ PCA
    scatter_pca(pca_tsne, y_train)
    plt.savefig('tsne_pca_all_train.png')