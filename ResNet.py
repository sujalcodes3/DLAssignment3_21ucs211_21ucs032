import numpy as npModel

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

def resnet_block(x, filters, kernel_size=3, stride=1):
    
    # skip connection(to where) for the resnet
    skip = x

    # First layer
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second layer
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # applying skip connections
    if stride != 1 or x.shape[-1] != skip.shape[-1]:
        skip = Conv2D(filters, 1, strides=stride)(skip)
        skip = BatchNormalization()(skip)
    x = Add()([x, skip])
    x = Activation('relu')(x)

    return x

def BuildResNet(input_shape, num_of_filters_1, num_of_filters_2):
    input_tensor = layers.Input(shape=input_shape)

    # First Layer
    x = Conv2D(num_of_filters_1, 3, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 16 Filter ResNet block
    x = resnet_block(x, num_of_filters_1)
    
    # 32 Filter ResNet block
    # we would downsample here with taking stride as 2 
    x = resnet_block(x, num_of_filters_2, stride=2)

    # Global average pooling and dense layer for classification
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

# Define the input shape and number of classes
input_shape = (32, 32, 3)
num_classes = 10

# Build the model with 16 filters in the first block and 32 filters in the second block
ResNetModel = BuildResNet(input_shape, num_of_filters_1=16, num_of_filters_2=32)

# Using Adam Optimizer
opt = Adam(lr=0.0005)

ResNetModel.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = ResNetModel.fit(x_train
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

