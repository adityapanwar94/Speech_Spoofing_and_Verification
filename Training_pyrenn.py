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
import os
from scipy.io import wavfile

# reading the source and target audio files
sourcepath = "/Users/arvindkumar/Desktop/eYSIP-2017/Datasets/arctic/source"
targetpath = "/Users/arvindkumar/Desktop/eYSIP-2017/Datasets/arctic/target"

source = os.listdir(sourcepath)
target = os.listdir(targetpath)
# print(source)  # Sometimes some invisible files might creep in.
# print(target)
source = source[0:1]
target = target[0:1]

frameLength = 1024
overlap = 0.25
hop_length = frameLength * overlap
net = pyrenn.CreateNN([26, 30, 30, 26])
order = 25
alpha = 0.41
gamma = -0.35

for sourcefile, targetfile in zip(source, target):
    print(sourcefile, targetfile)
    sr, sx = wavfile.read(sourcepath + "/" + sourcefile)
    sourceframes = librosa.util.frame(sx, frame_length=frameLength,  # framing the source audio
                                      hop_length=hop_length).astype(np.float64).T
    sourceframes *= pysptk.blackman(frameLength)  # windowing
    sourcemcepvectors = np.apply_along_axis(pysptk.mcep, 1, sourceframes, order,
                                            alpha)  # extract MCEPs of the source frames
    sr, tx = wavfile.read(targetpath + "/" + targetfile)
    targetframes = librosa.util.frame(tx, frame_length=frameLength,  # framing the target audio
                                      hop_length=hop_length).astype(np.float64).T
    targetframes *= pysptk.blackman(frameLength)  # windowing
    targetmcepvectors = np.apply_along_axis(pysptk.mcep, 1, targetframes, order,
                                            alpha)  # extract mceps of target frames
    reslen = min(len(sourcemcepvectors), len(targetmcepvectors))
    transsourcemcepvectorsmod = np.empty([26, reslen])
    transtargetmcepvectorsmod = np.empty([26, reslen])
    transsourcemcepvectorsmod = np.transpose(sourcemcepvectors[0:reslen])
    transtargetmcepvectorsmod = np.transpose(targetmcepvectors[0:reslen])
    print(len(sourcemcepvectors), len(targetmcepvectors))
    # to find if there are any NANs in the MCEP vectors
    for i in range(len(sourcemcepvectors)):
        for j in range(len(sourcemcepvectors[i])):
            if np.isnan(sourcemcepvectors[i][j]):
                print("yes")
    for i in range(len(targetmcepvectors)):
        for j in range(len(targetmcepvectors[i])):
            if np.isnan(targetmcepvectors[i][j]):
                print("no")
    print("Before", net["w"])
    # training the neural network  # TODO
    net = pyrenn.train_LM(transsourcemcepvectorsmod, transtargetmcepvectorsmod, net, k_max=100, verbose=True, E_stop=5)
    print("After", net["w"])

# Saving Model
pyrenn.saveNN(net, 'pyrennweights_2.csv')
