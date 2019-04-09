import pickle
import os
import numpy as numpy
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras. callbacks import LearningRateScheduler


def  build_model (n_kernels, kernel_size, stride, n_dense):

    # instantiate model
    model = Sequential()

    # convolution 1st layer
    # building a 2D convolutional neural network with input: n digits with  (n x 16 x 16 x 1) tensors , activation on forward propogation
    # apply n_kernels convolution filters (desired feature maps) on a square convolution kernel size (kernel_size x kernel_size)
    model.add(Conv2D(n_kernels,  (kernel_size, kernel_size), activation = 'relu',  input_shape=(16, 16, 1) ))
    #Normalize the activations of the previous layer at each batch
    model.add(BatchNormalization())
    #  Strides of the 2D max pooling layer
    model.add(MaxPooling2D(pool_size=(stride, stride)))
    # apply a dropout rate of 25% on input
    model.add(Dropout(0.25))
    # Flatten the input 
    model.add(Flatten())

    # hidden layer
    # Dense on number of hidden nodes n_dense in a densley connected layer
    model.add(Dense(n_dense, activation='relu'))
    # apply a dropout rate of 50% on input
    model.add(Dropout(0.5))

    # output layer
    # Dense on number of hidden nodes -> 10 class classification
    model.add(Dense(10, activation='softmax'))
    # compiling the deep model using categorical corss-entropy, adam optimizer with learning rate = 10^-4
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001))

    return model



if __name__ == '__main__':

    # loading the data 
    usps_data = pickle.load(open('usps.pickle', 'rb'))

    # split the data into train,  validation and test sets
    X_train = usps_data['x']['trn']
    Y_train = usps_data['y']['trn']
    X_val = usps_data['x']['val']
    Y_val = usps_data['y']['val']
    X_test = usps_data['x']['tst']
    Y_test = usps_data['y']['tst']

    # fitting the deep model to the training data
    model = build_model (n_kernels = 8, kernel_size = 3, stride = 2, n_dense = 32)
    annealer = LearningRateScheduler (lambda x: 1e-3 * 0.9 ** x)
    history = model.fit(X_train, Y_train, epochs = 2, batch_size = 16, verbose = 2,   validation_data = ( X_val, Y_val), callbacks = [annealer])

    # evaluating the model
    trn_score = model.evaluate(X_train, Y_train, batch_size = 16)
    tst_score = model.evaluate(X_test, Y_test, batch_size = 16)

    # priting the erros and networks params
    print("train error : " + str(trn_score))
    print("test error :  " + str(tst_score))
    print("number of network params:  " + str(model.count_params()))

