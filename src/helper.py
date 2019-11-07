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
    
    all_sd_names = [sd for sd in os.listdir('MURA_images/train/XR_ELBOW') if
                    os.path.isdir(os.path.join('../MURA_images/train/XR_ELBOW', sd))]
    count_sd = len([sd for sd in os.listdir('MURA_images/train/XR_ELBOW') if
                    os.path.isdir(os.path.join('MURA_images/train/XR_ELBOW', sd))])
    list_files_elbow = [len(files) for r,d, files in os.walk('MURA_images/train/XR_ELBOW') if len(files) != 0]
    list_files_all = [len(files) for r,d, files in os.walk('MURA_images/train') if len(files) != 0]

    file_counter_elbow = list(Counter(list_files_elbow).items())
    file_counter_all = list(Counter(list_files_all).items())

    
    x = [file_counter_all[i][0] for i in range(len(file_counter_all))]
    y = [file_counter_all[i][1] for i in range(len(file_counter_all))]

    # fig, ax = plt.subplots(figsize=(12,12))
    # plt.bar(x, y)
    # plt.title('Count of Training Directories with X images')
    # plt.savefig('Count_all_directories')
    # ax

    # from bash
    # find . -maxdepth 1 -mindepth 1 -type d -exec sh -c 'echo "{} : $(find "{}" -type f | wc -l)" file\(s\)' \

    search_word_list = ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST']

    # for word in search_word_list:
    #     find_and_sep_data('data/original_paths/train_labeled_studies.csv', 'path', word, info=0)
    
    # for word in search_word_list:
    #     find_and_sep_data('data/original_paths/valid_labeled_studies.csv', 'path', word, info=1)

    # df['target'] = np.where(df['path'].str.contains('positive'), 1, 0)

    # for word in search_word_list:
    #     find_and_sep_data('data/original_paths/train_image_paths.csv', 'path', word, info=0)
    
    # for word in search_word_list:
    #     find_and_sep_data('data/original_paths/valid_image_paths.csv', 'path', word, info=1)

    y_neg = [2925, 3183, 1164, 4059, 673, 4211, 5769]
    y_pos = [2006, 1968, 661, 1484, 599, 4168, 3987]
    fig, ax = plt.subplots(figsize=(12,12))
    p1 = plt.bar(search_word_list, y_neg, )
    p2 = plt.bar(search_word_list, y_pos)
    plt.legend((p1, p2), ('Negative', 'Positive'))
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.title('Negative/Positive Images per Bone')
    plt.savefig('Negative_Positive_Images_per_Bone.png')