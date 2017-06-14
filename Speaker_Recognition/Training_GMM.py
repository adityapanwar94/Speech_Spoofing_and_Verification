"""
Speaker Training.
Input data format is modified for ELSDSR Dataset.

"""

import numpy as np
from scipy.io import wavfile
from hmmlearn.hmm import GMMHMM
from python_speech_features import mfcc
import pickle
import os

from Model import GMMModel

# TODO: Add 1 to this every time you train a new speaker. Update this in the Classifier Program also.
speaker_number = 11

# CREATING A NEW FILE.
model_filename = "gmodel"+str(speaker_number)
f = open(model_filename, "wb")

# PARAMETERS.
N = 3  # Number of States.
Mixtures = 128  # Number of Gaussian Mixtures.

# CREATING A LIST OF FILE NAMES  # If lots of data is available.
# speaker_name = "FAML"  #TODO: Fill Speaker's name.
# file_path = "/Users/..."+speaker_name  #TODO: Add File Path that leads to a folder containing speaker's name.
# file_names = os.listdir(file_path)
# training_test_split = 0.75  # TODO: Decide on the split.
# number_of_files = int(0.training_test_split * len(file_names))
# file_names = file_names[0:number_of_files]
file_names = ["FMEL_Sa.wav", "FMEL_Sb.wav", "FMEL_Sc.wav", "FMEL_Sd.wav", "FMEL_Se.wav", "FMEL_Sf.wav", "FMEL_Sg.wav"]

# READING INPUT & MFCC FEATURE EXTRACTION.
lengths = np.empty(len(file_names))
feature_vectors = np.empty([0, 13])
for i in range(len(file_names)):
    rate, x = wavfile.read(file_names[i])
    x = mfcc(x, samplerate=rate)
    lengths[i] = len(x)
    feature_vectors = np.concatenate((feature_vectors, x))
# print(np.mean(feature_vectors))
# print(lengths)

# INITIAL PARAMETERS FOR HMM.
# startprob = np.ones(N) * (10**(-30))  # Left to Right Model
# startprob[0] = 1.0 - (N-1)*(10**(-30))
# transmat = np.zeros([N, N])  # Initial Transmat for Left to Right Model
# for i in range(N):
#     for j in range(N):
#         transmat[i, j] = 1/(N-i)
# transmat = np.triu(transmat, k=0)
# transmat[transmat == 0] = (10**(-30))

# GMM-HMM MODEL FITTING.
model = GMMHMM(n_components=N, n_mix=Mixtures, covariance_type='diag')
model.fit(feature_vectors)

# STORING THE MODEL.
sample = GMMModel(model, "FMEL")
pickle.dump(sample, f)

# POSSIBLE EXTENSIONS.
# TODO: Plug in various Initialisation Values and use score method to find the best fit model.
# TODO: Try to improve the way in which multiple models are trained.
