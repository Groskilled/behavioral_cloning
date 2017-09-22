from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda 
from keras.layers.convolutional import Conv2D, MaxPooling2D, Cropping2D
from keras.optimizers import SGD
from scipy.misc import imresize

def nvidia(input_shape=(224,224,3)):
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (1, 1)), input_shape = input_shape))
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten(name='flatten'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))
    return model
