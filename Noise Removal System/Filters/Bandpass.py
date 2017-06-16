###################################################################################################
#
# Author: Yash Agrawal
# Project: Speech Verification and Spoofing
#
# This is the implementation of a BANDPASS filter which can be used in the noise removal block 
###################################################################################################


from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
import scipy.io.wavfile
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz


# Several flavors of bandpass FIR filters.

def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps


def bandpass_kaiser(ntaps, lowcut, highcut, fs, width):
    nyq = 0.5 * fs
    atten = kaiser_atten(ntaps, width / nyq)
    beta = kaiser_beta(atten)
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=('kaiser', beta), scale=False)
    return taps


def bandpass_remez(ntaps, lowcut, highcut, fs, width):
    delta = 0.5 * width
    edges = [0, lowcut - delta, lowcut + delta,
             highcut - delta, highcut + delta, 0.5 * fs]
    taps = remez(ntaps, edges, [0, 1, 0], Hz=fs)
    return taps


#MAIN CODE STARTS HERE


rate, x = scipy.io.wavfile.read('sample.wav')
rate_n, n = scipy.io.wavfile.read('noise.wav')

#plots the signal in time domain
plt.figure('original signal')
plt.plot(x)

#adding noise to the input signal
n = n[0:len(x)]
xn = x + (n/15)

#plots the signal with added noise in time domain
plt.figure('signal with added noise')
plt.plot(xn)

#fourier transform of the signal with added noise
fourier_xn = np.fft.rfft(xn)

'''
#plots the frequency responce of signal with added noise
plt.figure('frequency responce of signal with added noise')
plt.plot(abs(fourier_xn))
'''

# Sample rate and desired cutoff frequencies (in Hz).
fs = rate *2
lowcut = 300
highcut = 2000

ntaps = 128

taps_hamming = bandpass_firwin(ntaps, lowcut, highcut, fs=fs)

taps_kaiser10 = bandpass_kaiser(ntaps, lowcut, highcut, fs=fs, width=1.0)

remez_width = 1.0
taps_remez = bandpass_remez(ntaps, lowcut, highcut, fs=fs, width=remez_width)

w, h1 = freqz(taps_hamming, 1, worN=len(fourier_xn))

w1, h2 = freqz(taps_kaiser10, 1, worN=len(fourier_xn))

w2, h3 = freqz(taps_remez, 1, worN=len(fourier_xn))

freq = fourier_xn
filtered = freq*h1    #choosing the window, h1=> hamming, h2=> kaiser, h3=> remez

'''
#plots filtered frequency responce
plt.figure('frequency responce of filtered signal')
plt.plot(abs(filtered))
'''

#plots filtered time domain responce
plt.figure('after removing the noise')
filtered_time_domain = np.fft.irfft(filtered)
plt.plot(filtered_time_domain)

plt.show()
