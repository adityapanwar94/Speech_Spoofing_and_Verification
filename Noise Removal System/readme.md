This folder contains Python files, Input speech and Waveforms for Noise Removal System.

ISSUE(S) : fft calculations are not very accurate in Python. MATLAB based coding might improve system response.

Concept:

It is based on the idea that an average estimate of noise from speech signal can be used to remove similar noise in the speech signal.

STEPS: 
1. Band Pass Filtering.
2. Speech Input Partitioned into Frames / Buffers with a predefined amount of overlap.
3. Initial Noisy Frames are Windowed, Transformed to Frequency Domain (FFT) and Averaged.
4. Each Frame is Windowed, Transformed to Frequency Domain (FFT) and Subtracted by Average Noise.
5. Half Wave Rectification is performed. 6. Reconstruction through IFFT and Overlap-Add Method.
