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

def get_data_classes(path):
    """
    Labels all data with bone + pos/neg for class assignment
    for unsupervised learning on bones + class (positive/negative).
    """
    all_images_as_array=[]
    label=[]
    for filename in os.listdir(path):
        try:
            if re.match(r'ELBOWnegative',filename):
                label.append(0)
            elif re.match(r'ELBOWpositive',filename):
                label.append(1)
            elif re.match(r'FINGERnegative',filename):
                label.append(2)
            elif re.match(r'FINGERpositive',filename):
                label.append(3)
            elif re.match(r'FOREARMnegative',filename):
                label.append(4)
            elif re.match(r'FOREARMpositive',filename):
                label.append(5)
            elif re.match(r'HANDnegative',filename):
                label.append(6)
            elif re.match(r'HANDpositive',filename):
                label.append(7)
            elif re.match(r'HUMERUSnegative',filename):
                label.append(8)
            elif re.match(r'HUMERUSpositive',filename):
                label.append(9)
            elif re.match(r'SHOULDERnegative',filename):
                label.append(10)
            elif re.match(r'SHOULDERpositive',filename):
                label.append(11)
            elif re.match(r'WRISTnegative',filename):
                label.append(12)
            elif re.match(r'WRISTpositive',filename):
                label.append(13)
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

if __name__ == "__main__":

    # # create training data
    train_data = "data/train_images/PCA/all/"
    X_train, y_train = get_data_classes(train_data)

    name = 'all_classes'

    time_start = time.time()

    x_subset = X_train[0:20000]
    y_subset = y_train[0:20000]

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_train)

    print ('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))

    pca_df = pd.DataFrame(columns = ['pca1','pca2','pca3'])#,'pca4','pca5','pca6','pca7','pca8','pca9','pca10'])

    pca_df['pca1'] = pca_result[:,0]
    pca_df['pca2'] = pca_result[:,1]

    print ('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))
    print(f'Total variance explained by 2 components: {sum(pca.explained_variance_ratio_)}')
    
    # taking top two principal components
    top_two_comp = pca_df[['pca1','pca2']]
    Visualizing the PCA output
    scatter_pca(top_two_comp.values, y_train)
    plt.savefig('pca_2_comp_'+name+'.png')

    # T-SNE on all datapoints - takes forever
    tsne_all = TSNE(random_state=RS, verbose=1).fit_transform(x_subset)
    scatter_pca(tsne_all, y_subset)
    plt.savefig('tsne_all_train_'+name+'.png')
    time_start = time.time()
    
    # T-SNE w/ PCA 
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(x_subset)
    print ('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
    print ('Variance explained per principal component: {}'.format(pca_50.explained_variance_ratio_))
    print(f'Total variance explained by 2 components: {sum(pca_50.explained_variance_ratio_)}')
    pca_tsne = TSNE(random_state=RS, verbose=1).fit_transform(pca_result_50)
    
    # plot PCA w/ T-SNE
    scatter_pca(pca_tsne, y_subset)
    plt.legend()
    plt.savefig('tsne_pca_'+name+'.png')

    tsne_df = pd.DataFrame(columns = ['pca1','pca2','pca3'])

    tsne_df['pca1'] = pca_result_50[:,0]
    tsne_df['pca2'] = pca_result_50[:,1]
    tsne_df['pca3'] = pca_result_50[:,2]

    #3d plot of PCA w/ T-SNE
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=tsne_df["pca1"], 
        ys=tsne_df["pca2"], 
        zs=tsne_df["pca3], 
        c=y_subset, 
        cmap='tab10'
        )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.legend()
    plt.tight_layout()
    plt.savefig('3d_tsne_'+name+'.png')