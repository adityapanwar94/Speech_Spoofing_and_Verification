"""
This Program is used to convert one voice to another. It uses a simple method of reconstructing the speech signal.
Dataset used : arctic

Steps:
1. Feature Extraction.
2. Loading Trained Model.
3. Obtaining New MGCEP Features.
4. Reconstruction by Back Tracing how MGCEP is extracted.

Notes:
1. This program is made to convert one source speaker to only one target speaker. If more targets are trained,
some encoding must be done so that the correct model is retrieved.
2. Currently, trained only with 4 Short Training Examples. So, Model does not generalise well.

"""

import numpy as np
import pysptk
import librosa
import pyrenn
from scipy.io import wavfile

sourcepath = "/Users/arvindkumar/Desktop/eYSIP-2017/Datasets/arctic/source"
sourcefile = 'arctic_a0002.wav'

frameLength = 1024
overlap = 0.25
hop_length = frameLength * overlap
net = pyrenn.CreateNN([26, 30, 30, 26])
order = 25
alpha = 0.41
gamma = -0.35

# Loading pyrenn Model
net = pyrenn.loadNN('pyrennweights_2.csv')

# testing source file
sr, sx = wavfile.read(sourcepath + "/" + sourcefile)
l = len(sx)

# framing
sourceframes = librosa.util.frame(sx, frame_length=frameLength,
                                  hop_length=hop_length).astype(np.float64).T

# Windowing
sourceframes *= pysptk.blackman(frameLength)

# extract MCEPs
sourcemcepvectors = np.apply_along_axis(pysptk.mcep, 1, sourceframes, order, alpha)
mgc = np.empty([len(sourcemcepvectors), 26])
# provide the source MCEPs as input to the trained neural network which gives the target MCEPs
mgc = pyrenn.NNOut(sourcemcepvectors.transpose(), net).transpose()
mgc = mgc.copy(order="C")
# apply mgc2sp to get spectral estimation from MCEPs
logHmgc = np.apply_along_axis(pysptk.mgc2sp, 1, mgc, 0.41, 0.0, frameLength)
# inverse log operation to obtain audio in frequency domain
rawHmgc = np.exp(logHmgc).T
# Inverse fourier transform to convert to time domain
mgcRecover = librosa.core.istft(rawHmgc, hop_length, frameLength,
                                pysptk.blackman(frameLength))
librosa.output.write_wav("Pyrenn_out.wav", mgcRecover, sr)
