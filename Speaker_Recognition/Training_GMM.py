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
from os import listdir

from Model import GMMModel

""" CREATING FILE TO STORE LEARNED MODEL """

speaker_number = 10  # TODO: Add 1 before training a new speaker. Update this in the Classifier Program also.
model_filename = "gmodel"+str(speaker_number)
f = open(model_filename, "wb")

""" PARAMTERS """

N = 3  # Number of States.
Mixtures = 128  # Number of Gaussian Mixtures.

""" CREATING LIST OF TRAINING FILE NAMES """

# os.listdir(filepath) can also be used.  # TODO: Update file_names before training a new speaker.
file_names = ["FAML_Sa.wav", "FAML_Sb.wav", "FAML_Sc.wav", "FAML_Sd.wav", "FAML_Se.wav", "FAML_Sf.wav", "FAML_Sg.wav"]

""" FEATURE EXTRACTION """

lengths = np.empty(len(file_names))
feature_vectors = np.empty([0, 13])
for i in range(len(file_names)):
    rate, x = wavfile.read(file_names[i])
    x = mfcc(x, samplerate=rate)
    lengths[i] = len(x)
    feature_vectors = np.concatenate((feature_vectors, x))
# print(feature_vectors.shape)

""" MODEL WITH INITIAL PARAMETERS """

# Might provide better results in some cases.

# startprob = np.ones(N) * (10**(-30))  # Left to Right Model
# startprob[0] = 1.0 - (N-1)*(10**(-30))
# transmat = np.zeros([N, N])  # Initial Transmat for Left to Right Model
# for i in range(N):
#     for j in range(N):
#         transmat[i, j] = 1/(N-i)
# transmat = np.triu(transmat, k=0)
# transmat[transmat == 0] = (10**(-30))
# model = GMMHMM(n_components=N, n_mix=Mixtures, covariance_type='diag', init_params="mcw")
# model.startprob_ = startprob
# model.transmat_ = transmat

""" MODEL WITHOUT INITIAL PARAMETERS """

model = GMMHMM(n_components=N, n_mix=Mixtures, covariance_type='diag')

""" MODEL FITTING """

model.fit(feature_vectors)

""" STORING THE MODEL """

sample = GMMModel(model, "FAML")  # TODO: Change Name as well.
pickle.dump(sample, f)

"""" FUTURE EXTENSIONS """

# TODO: Use score method to evaluate the model and run multiple iterations until best fit.
# TODO: Create a loop so that multiple speakers can be trained in one run.
