from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda 
from keras.layers.convolutional import Conv2D, MaxPooling2D, Cropping2D
from keras.optimizers import SGD
from scipy.misc import imresize

def nvidia(input_shape=(224,224,3)):
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (1, 1)), input_shape = input_shape))
    model.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten(name='flatten'))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(80, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model
