import os
import re
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM

OUTPUT_FNAME = 'RESULT.png'

#DATA_DIR_C1 = './Data/Halo-NoHalo/Halo/'
#DATA_DIR_C2 = './Data/Halo-NoHalo/NoHalo/'

#IMG_SIZE_W = 610
#IMG_SIZE_L = 610

#DATA_DIR_C1 = './Data/Halo-NoHalo_small/Halo/'
#DATA_DIR_C2 = './Data/Halo-NoHalo_small/NoHalo/'

DATA_DIR_C1 = 'D:/Playground/dl-neurotracker/Data/Halo-NoHalo_s32/Halo/'
DATA_DIR_C2 = 'D:/Playground/dl-neurotracker/Data/Halo-NoHalo_s32/NoHalo/'

IMG_SIZE_W = 32
IMG_SIZE_L = 32

def load_data():
    data = []
    nbImages = 0
    for fname in tqdm(os.listdir(DATA_DIR_C1)):
        label = 1
        img = Image.open(DATA_DIR_C1 + fname)
        data.append([np.array(img), np.array(label)])
        nbImages+=1

    for fname in tqdm(os.listdir(DATA_DIR_C2)):
        label = 0
        img = Image.open(DATA_DIR_C2 + fname)
        data.append([np.array(img), np.array(label)])
        nbImages+=1

    return data

data = load_data()
print(len(data))

data = shuffle(data)
data_train = data[0:6000]
data_test = data[6000:]
print(len(data_train))
print(len(data_test))

X_train = np.array([d[0] for d in data_train])
Y_train = np.array([d[1] for d in data_train])

X_test = np.array([d[0] for d in data_test])
Y_test = np.array([d[1] for d in data_test])

print('X_train: ' + str(len(X_train)))
print('Y_train: ' + str(len(Y_train)))
print('X_test: ' + str(len(X_test)))
print('Y_test: ' + str(len(Y_test)))

data = []
data_train = []
data_test = []

X_train = X_train.reshape(X_train.shape[0], 1, IMG_SIZE_W, IMG_SIZE_L)
X_test = X_test.reshape(X_test.shape[0], 1, IMG_SIZE_W, IMG_SIZE_L)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('Input shape: ' + str(X_train[0].shape))
print('Output shape: ' + str(Y_train[0].shape))
print('X_train shape: ' + str(X_train.shape))
print('Y_train shape: ' + str(Y_train.shape))

from keras.utils import np_utils

# Preprocess class labels
Y_train = np_utils.to_categorical(Y_train, 2)
Y_test = np_utils.to_categorical(Y_test, 2)

# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# 7. Define model architecture
model = Sequential()
 
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,IMG_SIZE_W, IMG_SIZE_L), data_format="channels_first"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
 
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# 9. Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=16, epochs=20, verbose=1)
 
# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=1)

model.save('LowRes.h5')

