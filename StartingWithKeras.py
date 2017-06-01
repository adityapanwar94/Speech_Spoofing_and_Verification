###################################################################################################
#
# Author: Yash Agrawal
# Project: Speech Verification and Spoofing
#
# This code is just to get the hang of keras.
# This code demonstrates a 3 input XOR gate trained using keras ML library.
###################################################################################################
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#This is the input data for training
training_data = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], "float32")

#This is the output data for training
target_data = np.array([[0], [1], [1], [0], [1], [0], [0], [1]], "float32")

#Here we start designing our neural network layer by layer.
#Our neural network has 3 input neurons, 40 and 20 neurons in successive hidden layer.
#Then there is one out neuron to binary output.
model = Sequential()
model.add(Dense(40, input_dim=3, activation='relu'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

#Here we compile the whole network, think of it like connecting everything
#And we also set a loss function and optimizer.
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

#This is the training part.
model.fit(training_data, target_data, nb_epoch=1000)

#We get the output for the same training data for the trained network.
print(model.predict(training_data).round())
