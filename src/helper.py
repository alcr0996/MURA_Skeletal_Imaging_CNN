import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
from PIL import Image

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
# old_im = Image.open("path/to/old/image.png")
# if old_im.mode != 'RGB':
#     old_im = old_im.convert('RGB')
# pixels = old_im.load()
# width, height = old_im.size[0], old_im.size[1]
# rows_to_remove = find_rows_with_color(pixels, width, height, (0, 0, 0)) #Remove black rows
# new_im = Image.new('RGB', (width, height - len(rows_to_remove)))
# pixels_new = new_im.load()
# rows_removed = 0
# for y in xrange(old_im.size[1]):
#     if y not in rows_to_remove:
#         for x in xrange(new_im.size[0]):
#             pixels_new[x, y - rows_removed] = pixels[x, y]
#     else:
#         rows_removed += 1
# new_im.save("path/to/new/image.png")


if __name__ == "__main__":

    all_sd_names = [sd for sd in os.listdir('MURA_images/train/XR_ELBOW') if
                    os.path.isdir(os.path.join('../MURA_images/train/XR_ELBOW', sd))]
    count_sd = len([sd for sd in os.listdir('MURA_images/train/XR_ELBOW') if
                    os.path.isdir(os.path.join('MURA_images/train/XR_ELBOW', sd))])
    list_files_sd = [len(files) for r,d, files in os.walk('MURA_images/train/XR_ELBOW') if len(files) != 0]
    file_counter = list(Counter(list_files_sd).items())
    
    x = [file_counter[i][0] for i in range(len(file_counter))]
    y = [file_counter[i][1] for i in range(len(file_counter))]

    fig, ax = plt.subplots(figsize=(12,12))
    plt.bar(x, y)
    # fig, ax = plt.subplts(figsize=(12,12))
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

          