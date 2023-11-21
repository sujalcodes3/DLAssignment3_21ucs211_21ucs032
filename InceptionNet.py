import numpy as np

from keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K

from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import plot_model
from tensorflow.keras import layers, Model

import tensorflow as tf
import keras
from keras.layers import *
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import AveragePooling2D
from keras.layers import UpSampling2D
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt


def inception_module(x, filters):
  """
  Inception module of the InceptionNet
  """
  tower_1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
  tower_1 = Conv2D(filters[1], (3, 3), padding='same', activation='relu')(tower_1)

  tower_2 = Conv2D(filters[2], (1, 1), padding='same', activation='relu')(x)
  tower_2 = Conv2D(filters[3], (5, 5), padding='same', activation='relu')(tower_2)

  tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
  tower_3 = Conv2D(filters[4], (1, 1), padding='same', activation='relu')(tower_3)

  output = Concatenate(axis=-1)([tower_1, tower_2, tower_3])
  return output

def InceptionNet(input_shape, num_classes):
  """
  InceptionNet architecture using functional API.
  """
  input_tensor = Input(shape=input_shape)

  x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_tensor)
  x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
  x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
  x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
  x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

  x = inception_module(x, [16, 16, 16, 16, 16])
  x = inception_module(x, [32, 32, 32, 32, 32])

  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.4)(x)
  x = Dense(num_classes, activation='softmax')(x)

  model = Model(inputs=input_tensor, outputs=x, name='InceptionNet')
  return model

InceptNetModel = InceptionNet((32, 32, 3), 10)
opt = Adam(lr=0.0005)
InceptNetModel.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = InceptNetModel.fit(x_train
          , y_train
          , batch_size=32
          , epochs=10
          , shuffle=True
          , validation_data = (x_test, y_test))

# Accuracy vs Epoch   
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Loss vs Epoch   
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

