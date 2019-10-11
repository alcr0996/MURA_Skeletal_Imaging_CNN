import os
os.environ['TF_KERAS']='1'

import numpy as np
np.random.seed(1337)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras_radam import RAdam

datagen = ImageDataGenerator(
        rotation_range = 90,
        width_shift_range = 0.5,
        height_shift_range = 0.5,
        shear_range = 0.5,
        zoom_range = 0.5,
        horizontal_flip = True,
        vertical_flip = False,
        fill_mode = 'nearest'
)

# img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='preview', save_prefix='elbow', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely

model = Sequential()
model.add(Conv2D(32, (1,1), input_shape=(3, 150, 150), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

model.add(Conv2D(64, (1,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64), activation = 'relu')
model.add(Dropout(0.5))

model.add(Dense(1), activation='sigmoid')


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '../data/train/XR_ELBOW',  # this is the target directory
        target_size=(50, 50),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/cat_dog/cat_dog',
        target_size=(50, 50),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('first_try.h5')  # always save your weights after training or during training