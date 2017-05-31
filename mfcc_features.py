from python_speech_features import mfcc
import numpy as np
from scipy.io import wavfile as sp
import matplotlib.pyplot as plt

rate, x = sp.read('sample.wav')
features = mfcc(x,rate)

print(features)
print(len(features))
print(len(features[0]))

'''
####################################################
PROBLEMS BEING FACED:

The code give output as the features of one particular audio file.

It is noted that the features extracted will have 13 features in each frame and number of frames will vary
depending upon the length of the audio file and also the rate of sampling.

The question arises as to how exactly this data need to be fed into an LSTM recurrent neural network.
Will the model have 13 blocks at the input and if yes, how many blocks at the output.

If we are simply training the model to recoganize "yes" or "no" and if we have 400 training data(mfcc features) for each of the cases, how to go about it.

####################################################
'''

