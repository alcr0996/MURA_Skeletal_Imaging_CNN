import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import Counter
import pandas as pd
from PIL import Image
plt.rcParams.update({'font.size': 20})
# plt.style.use('ggplot')

def find_and_sep_data(data, column, search_word, info = 0):
    '''
    Search files by row for specific data.
    data: path to data
    column: column or list of columns you want to specifically look at
    search_word: string you are searching for
    type: whether the information is for validation or training
            training = 0, validation = 1
    '''

    df = pd.read_csv(data, names=['path', 'target'])
    
    df_new = df[df[column].str.contains(search_word) == True]

    if info == 0:
        info = 'train_dir_paths'
    else:
        info = 'valid_dir_paths'

    df_new.to_csv('data/'+info+'/'+search_word+'_image_paths.csv')


def find_rows_with_color(pixels, width, height, color):
    '''
    Remove rows from an image where every pixel is black
    '''
    rows_found=[]
    for y in xrange(height):
        for x in xrange(width):
            if pixels[x, y] != color:
                break
        else:
            rows_found.append(y)
    return rows_found

if __name__ == "__main__":