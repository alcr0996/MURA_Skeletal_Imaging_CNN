import shutil
import os
import os.path
import pdb

def copy_move_files(path, target_dir):
    # pdb.set_trace()
    word_list = ['ELBOW', 'FINGER', 'HAND', 'WRIST', 'FOREARM', 'HUMERUS', 'SHOULDER'] 
    for word in word_list:
        i = 1
        directory_list = [x[0] for x in os.walk(path+'XR_'+word)]
        for directory in directory_list:
            if 'positive' in directory:
                for root, direct, files in os.walk(directory):
                    for f in files:
                        shutil.copy(root+'/'+f, target_dir+word+'/positive/image'+str(i)+'.png')
                        i += 1
                        print(i)
            elif 'negative' in directory:
                for root, direct, files in os.walk(directory):
                    # print(root)
                    # print(dirs)
                    # print(files)
                    for f in files:
                        # base, extension = os.path.splitext(filename)
                        shutil.copy(root+'/'+f, target_dir+word+'/negative/image'+str(i)+'.png')
                        i += 1
                        print(i)
            else:
                continue
if __name__ == "__main__":
    copy_move_files('MURA_images/train/', 'data/train_images/')
    copy_move_files('MURA_images/valid/', 'data/valid_images/')