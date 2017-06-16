"""
NOISE REMOVAL USING SPECTRAL SUBTRACTION FOR SPEECH RECOGNITION

Author(s) : Arvind Kumar
Date      : 26th May 2017

System removes basic noise from the input speech signal using Spectral Subtraction

Concept:
1. A Band Pass Filter is implemented to reject any signal component outside the range of 20 to 4000 Hz
since that is where all of speech activity lies.
2. Spectral Subtraction method is used to remove noise content from the speech signal.
STEPS:
    1. Speech Input Partitioned into Frames / Buffers with a predefined amount of overlap.
    2. Initial Noisy Frames are Windowed, Transformed to Frequency Domain (FFT) and Averaged
    3. Each Frame is Windowed, Transformed to Frequency Domain (FFT) and Subtracted by Average Noise.
    4. Half Wave Rectification is performed
    5. Reconstruction through IFFT and Overlap-Add Method

Notes:
1. A brief period of silence (around 0.5 seconds) before speech makes the system better at removing noise.
2. Assumes No Abrupt Noise during speech, echos or reverberation. These are considered features
and will be classified so by the Neural Network.
3. Some lines of code are commented so that they can be used while testing.

"""

import numpy as np
from scipy.io import wavfile as sp
import matplotlib.pyplot as plt

# Getting Input Speech
rate, x = sp.read('Input.wav')  # Audio File Named Input.wav must be in the same Directory
# x = np.ones(16)

# Spectral Subtraction Parameters
Buffer_Size = 256  # int(rate*0.025)  # 1 Buffer is around 20 to 30 ms in length.
# print(Buffer_Size)
Overlap = 0.5
noisy_frames = 10  # Value is Adjustable
beta = 1

BO = Buffer_Size * Overlap

# Pad zeros until it becomes a multiple of Buffer Size
while len(x) % Buffer_Size != 0:
    x = np.append(x, 0)

time_axis = np.arange(len(x))
y = np.zeros(len(x))
# Plotting the Input Data
plt.figure(1)
plt.plot(time_axis, x)
plt.show()

# Make a list of noisy frames
noisy_frame_list = np.arange(1, noisy_frames+1, 1)  # This can be trained as well.
no_of_noisy_frames = len(noisy_frame_list)

# Finding the Last Frame
last_frame = int(len(x) / BO)
# print(last_frame)


# Function To Get a Frame from Frame Number
def get_frame(n):
    segment = np.zeros(Buffer_Size)
    v = int((n-1) * BO)
    for q in np.arange(v, v+Buffer_Size, 1):
        segment[q-v] = x[q]
    return segment


# Average Noise Estimate
def avg_noise():
    if no_of_noisy_frames == 0:
        return np.zeros(Buffer_Size)
    else:
        f_n = np.zeros(Buffer_Size)
        for b in noisy_frame_list:
            n = np.array(get_frame(b))
            n = np.multiply(n, np.hanning(Buffer_Size))
            f_n = np.add(f_n, np.fft.fft(n))
        mu_n = f_n / no_of_noisy_frames
        return abs(mu_n)

# Spectral Subtraction
mu = avg_noise()
# print(mu)
for i in np.arange(1, last_frame, 1):
    number = int((i-1)*BO)
    a = np.array(get_frame(i))
    # print(a)
    # Windowing Using a Hanning Window
    w_a = np.multiply(a, np.hanning(Buffer_Size))
    # Converting to Frequency Domain
    f_a = np.fft.fft(w_a)
    # print(f_a)
    mag_a = abs(f_a)
    # print(mag_a)
    ph_a = np.angle(f_a)
    # Half Wave Rectification
    s_a = np.zeros(len(a))
    for l in np.arange(0, len(mag_a), 1):
        if mag_a[l] >= mu[l]:
            s_a[l] = mag_a[l] - beta*mu[l]
        else:
            s_a[l] = 0
    # print(s_a)
    # Combining Phase
    sp_a = np.multiply(s_a, np.exp(1j*ph_a))
    # print(sp_a)
    # Converting back to Time Domain
    clear_a = np.fft.ifft(sp_a)
    # print(clear_a)
    # print(clear_a)
    # Reconstruction using Overlap - Add Method
    for index in np.arange(number, number + Buffer_Size, 1):
        y[index] = y[index] + np.real(clear_a[index-number])
        # print(y)
# print(y)
sp.write('Output.wav', 44100, y)
# Plotting Output
plt.figure(2)
plt.plot(time_axis, y)
plt.show()