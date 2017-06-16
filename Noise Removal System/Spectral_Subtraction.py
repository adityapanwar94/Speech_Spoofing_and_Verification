import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from python_speech_features import sigproc

# Parameters.
# winsize = 0.025
# winstep = 0.010
no_of_noisy_frames = 10

# Getting Input Speech.
rate, noisy_speech = wavfile.read('E1.wav')  # Audio File Named Input.wav must be in the same Directory

# Plot Input.
plt.figure(1)
plt.plot(range(len(noisy_speech)), noisy_speech)
plt.show()

# Framing.
frame_list = sigproc.framesig(noisy_speech, frame_len=512, frame_step=256, winfunc=np.hamming)
# print(frame_list)

# Calculating Average Noise.
noisy_frame_list = frame_list[0:no_of_noisy_frames]
noisy_frame_magspec = np.empty([no_of_noisy_frames, 512])
for i in range(len(noisy_frame_list)):
    noisy_frame_magspec[i] = np.fft.fft(noisy_frame_list[i])
mu_array = noisy_frame_magspec.mean(0)

# Subtraction.
for j in range(len(frame_list)):
    l = 512
    frame_freq = np.fft.fft(frame_list[j])
    frame_mag = np.absolute(frame_freq)
    frame_phase = np.angle(frame_freq)
    frame_subtracted = np.empty(512)
    for i in range(512):
        if frame_mag[i] > mu_array[i]:
            frame_subtracted[i] = frame_mag[i] - mu_array[i]
        else:
            frame_subtracted[i] = 0
    frame_spectra = np.multiply(frame_subtracted, np.exp(1j*frame_phase))
    frame_list[j] = np.fft.ifft(frame_spectra)

# Overlap Add.
output_signal = sigproc.deframesig(frame_list, siglen=len(noisy_speech), frame_len=512, frame_step=256)

# Write to Output File.
wavfile.write('Output.wav', rate=rate, data=output_signal)

# Plot Output.
plt.figure(2)
plt.plot(range(len(noisy_speech)), output_signal)
plt.show()



