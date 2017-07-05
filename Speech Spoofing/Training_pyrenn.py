"""
Speech Spoofing: Training.
Program to map Source MGCEP Features to Target MGCEP Features.

There are 2 main steps
1. Feature Extraction.
2. Recurrent Neural Network Training.

Feature Extraction:
1. All input files are created in a list.
2. Decisions made on various parameters given below.
3. MGCEP features are extracted and a single feature vector array is created.
4. It is Pre processed in order to be accepted by RNN Model.

Neural Network:
1. A Recurrent Neural Network is created using pyrenn Package.
Architecture : 2 Hidden Layers with 30 cells are created.
2. Model is trained.
3. Trained Model is saved.

Notes:
1. It is assumed that the input data is time aligned. If not, Dynamic Time Warping (DTW) is necessary.
"""

import numpy as np
import pysptk
import librosa
import pyrenn
from scipy.io import wavfile
import os

sourcefile = "source_train.wav"
targetfile = "target_train.wav"

# Parameters.
frameLength = 1024
overlap = 0.25
hop_length = frameLength * overlap
order = 25
alpha = 0.42
gamma = -0.35

# Feature Extraction.
sr, sx = wavfile.read(sourcefile)
sourceframes = librosa.util.frame(sx, frame_length=frameLength,  # framing the source audio
                                  hop_length=hop_length).astype(np.float64).T
sourceframes *= pysptk.blackman(frameLength)  # windowing
sourcemcepvectors = np.apply_along_axis(pysptk.mcep, 1, sourceframes, order,
                                        alpha)  # extract MCEPs of the source frames
sr, tx = wavfile.read(targetfile)
targetframes = librosa.util.frame(tx, frame_length=frameLength,  # framing the target audio
                                  hop_length=hop_length).astype(np.float64).T
targetframes *= pysptk.blackman(frameLength)  # windowing
targetmcepvectors = np.apply_along_axis(pysptk.mcep, 1, targetframes, order,
                                        alpha)  # extract mceps of target frames

# Normalising for feeding into RNN.
norm = min(len(sourcemcepvectors), len(targetmcepvectors))
transsourcemcepvectorsmod = np.transpose(sourcemcepvectors[0:norm])
transtargetmcepvectorsmod = np.transpose(targetmcepvectors[0:norm])

# Training Model.
net = pyrenn.CreateNN([order+1, order+5, order+5, order+1])
net = pyrenn.train_LM(transsourcemcepvectorsmod, transtargetmcepvectorsmod, net, k_max=100, verbose=True, E_stop=5)

# Saving Model.
pyrenn.saveNN(net, 'pyrennweights_2.csv')
