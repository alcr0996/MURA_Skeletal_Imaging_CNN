from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import PReLU, LeakyReLU
import numpy as np

# dimensions of our images.
img_width, img_height = 224, 224
train_data_dir = 'data/train_images/FOREARM'
validation_data_dir = 'data/valid_images/FOREARM'
nb_train_samples = 1830
nb_validation_samples = 301
epochs = 30
batch_size = 25


model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224, 224, 3)))
model.add(Convolution2D(64, (3, 3), activation=LeakyReLU()))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation=LeakyReLU()))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation=LeakyReLU()))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation=LeakyReLU()))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation=LeakyReLU()))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation=LeakyReLU()))
model.add(Dropout(0.5))
model.add(Dense(4096, activation=LeakyReLU()))
model.add(Dropout(0.5))
model.add(Dense(1000, activation=LeakyReLU()))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
opt = Adam(learning_rate = 0.0005)
model.compile(loss='binary_crossentropy',optimizer='adam',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

#model.load_weights('data/model_weights/best_model_so_far.h5')         
model_name = 'vg_244_model'
model.save_weights('data/model_weights/'+model_name+'.h5')
model.save('data/models/'+model_name+'.h5')


 # this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
        )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode='binary')


validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode='binary')

history = model.fit_generator(
train_generator,
steps_per_epoch= nb_train_samples // batch_size,
epochs=epochs,
validation_data=validation_generator,
validation_steps=nb_validation_samples // batch_size
#     callbacks = [callback]
)

validation_generator.class_indices

# plot loss during training
fig, ax = plt.subplots(2, figsize = (12, 8))
ax[0].set_title('Loss')
ax[0].set_xticks(range(1,31,1))
ax[0].plot(history.history['loss'], label='train')
ax[0].plot(history.history['val_loss'], label='test')
ax[0].legend()

# plot accuracy during training
ax[1].set_xticks(range(1,31,1))
ax[1].set_title('Accuracy')
ax[1].plot(history.history['accuracy'], label='train')
ax[1].plot(history.history['val_accuracy'], label='test')
ax[1].legend()

plt.savefig('vg_224.png')

# Confusion Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, 301 // batch_size+1)
y_pred = np.where(Y_pred>=.50, 1, 0)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))

print('Classification Report')
target_names = ['positive', 'negative']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

# ROC

fpr, tpr, thresholds = roc_curve(validation_generator.classes, Y_pred)
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
plt.savefig("vg_ROC_224.png")