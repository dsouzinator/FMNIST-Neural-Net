### Created by https://github.com/dsouzinator on github
### Please keep these set of comments for future use

from __future__ import print_function
import numpy as np  # linear algebra
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.layers import LeakyReLU
from keras import backend as K
from keras.models import Model
from keras.datasets import fashion_mnist
import mnist_reader
import matplotlib.pylab as plt

from keras.datasets import fasion_mnist
((X_train, y_train), (X_test, y_test)) = fashion_mnist.load_data()


X_train = X_train.reshape([-1, 28, 28, 1])
X_test = X_test.reshape([-1, 28, 28, 1])
print(X_train.shape)

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train, 10)

y_test = to_categorical(y_test, 10)
img_x, img_y = 28, 28
input_shape = (img_x, img_y, 1)


def fashion_model(input_shape):
    fashionModel = Sequential()
     fashionModel.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), name='conv0', padding='same', input_shape=input_shape))
     fashionModel.add(BatchNormalization(axis=3, name='bn0'))
     # fashionModel.add(Activation('relu'))
     fashionModel.add(LeakyReLU(alpha=0.1))

     fashionModel.add(Conv2D(64, (7, 7), strides=(1, 1), name='conv1', padding='same'))
     fashionModel.add(MaxPooling2D((2, 2), name='max_pool1'))
     fashionModel.add(Dropout(0.25))
     fashionModel.add(BatchNormalization(axis=3, name='bn1'))
     # fashionModel.add(Activation('relu'))
     fashionModel.add(LeakyReLU(alpha=0.1))

     fashionModel.add(Conv2D(200, (7, 7), strides=(2, 2), name='conv2b', padding='same'))
     fashionModel.add(Conv2D(64, (7, 7), strides=(2, 2), name='conv2c', padding='same'))
     fashionModel.add(Conv2D(32, (5, 5), strides=(2, 2), name='conv3', padding='same'))
     fashionModel.add(MaxPooling2D((2, 2), name='max_pool2'))
     fashionModel.add(Dropout(0.3))
     fashionModel.add(BatchNormalization(axis=3, name='bn2'))
     # fashionModel.add(Activation('relu'))
     fashionModel.add(LeakyReLU(alpha=0.1))

     fashionModel.add(Flatten())
     fashionModel.add(Dense(250, activation='linear', name='fc1d'))
     fashionModel.add(LeakyReLU(alpha=0.1))
     fashionModel.add(Dropout(0.25))
     fashionModel.add(Dense(500, activation='linear', name='fc2b'))
     fashionModel.add(LeakyReLU(alpha=0.1))
     fashionModel.add(Dropout(0.25))
     fashionModel.add(Dense(250, activation='linear', name='f2c'))
     fashionModel.add(LeakyReLU(alpha=0.1))
     fashionModel.add(Dropout(0.4))
     fashionModel.add(Dense(10, activation='softmax', name='fc3a'))

     return fashionModel


##############################################################33
fashionModel = fashion_model(input_shape)


fashionModel.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
fashionModel.fit(X_train, y_train, epochs=60, batch_size=64)

preds = fashionModel.evaluate(X_test, y_test, batch_size=16)

print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
fashionModel.save('new_with_nn_rerun_drop.h5')
fashionModel.save_weights('new_with_nn_weights1b_rerun_drop.h5')
