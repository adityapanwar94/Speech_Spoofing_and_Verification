"""
Speaker Training.

1. One GMM-HMM Model is built for every speaker using the Training Data.
2. MFCC Vectors are used as features. (13 features per frame)
3. Model is fit using Expectation-Maximization(EM) Algorithm.
4. Obtained Model is stored into a distinct file using pickle.

NOTE:
1. Minor changes must be in case VCTK-Database is used. These are commented inside the code.

"""

import numpy as np
from scipy.io import wavfile
from hmmlearn.hmm import GMMHMM
from python_speech_features import mfcc
import pickle
import os

from Model import GMMModel

# PARAMETERS.
N = 3  # Number of States.
Mixtures = 128  # Number of Gaussian Mixtures.

# CREATING A NEW FILE # If VCTK-Corpus is used.
# speaker_number = 225
# serial_number = 1  # TODO: Add 1 before training a new speaker. Update this in the Classifier Program also.
# model_filename = "vmodel"+str(serial_number)

# CREATING A NEW FILE. # If ELSDSR is used.
speaker_number = 11  # TODO: Add 1 before training a new speaker. Update this in the Classifier Program also.
model_filename = "gmodel"+str(speaker_number)

f = open(model_filename, "wb")

# CREATING A LIST OF FILE NAMES # If VCTK-Corpus is used.
# file_path = "/Users/arvindkumar/Desktop/Speaker Recognition/VCTK-Corpus/VCTK-Corpus/wav48/p"+str(speaker_number)
# file_names = os.listdir(file_path)
# training_test_split = 0.75  # TODO: Decide on the split.
# number_of_files = int(0.training_test_split * len(file_names))
# file_names = file_names[0:number_of_files]

# CREATING A LIST OF FILE NAMES # If ELSDSR is used.
file_names = ["FMEL_Sa.wav", "FMEL_Sb.wav", "FMEL_Sc.wav", "FMEL_Sd.wav", "FMEL_Se.wav", "FMEL_Sf.wav", "FMEL_Sg.wav"]

# READING INPUT & MFCC FEATURE EXTRACTION.
lengths = np.empty(len(file_names))
feature_vectors = np.empty([0, 13])
for i in range(len(file_names)):
    # rate, x = wavfile.read(file_path+str(speaker_number)+"/"+file_names[i])  # If VCTK-Corpus is used.
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

# STORING THE MODEL. # If VCTK-Corpus is used.
# sample = GMMModel(model, "p"+str(speaker_number))

# STORING THE MODEL. If ELSDSR is used.
sample = GMMModel(model, "FMEL")

pickle.dump(sample, f)

# POSSIBLE EXTENSIONS.
# TODO: Plug in various Initialisation Values and use score method to find the best fit model.
# TODO: Try to improve the way in which multiple models are trained.
