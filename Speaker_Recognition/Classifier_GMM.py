"""
Classifier Program.
Input data format is modified for ELSDSR Dataset.

"""

from scipy.io import wavfile
from python_speech_features import mfcc
import pickle
import numpy as np

# PARAMETERS
# N = 3   # Number of Hidden States.
# n_mix = 128
# threshold = 0.000000  # Could be used to detect a new speaker...
no_of_speakers = 10  # Number of Speakers in the Training Set.

# INPUT.
test_speech1 = 'FDHH_Sr26.wav'

# EXTRACTING MFCC FEATURES.
test_speech_name = test_speech1[0:4]
rate, speech1 = wavfile.read(test_speech1)
feature_vectors1 = mfcc(speech1, samplerate=rate)

# COMPUTING LOG PROBABILITY VECTOR FOR EVERY MODEL.
probability_vector = np.empty(no_of_speakers)

for i in range(no_of_speakers):
    model_filename = "gmodel"+str(i+1)
    sample = pickle.load(open(model_filename, "rb"))

    # RUN FORWARD ALGORITHM TO RETURN PROBABILITY.
    p1 = sample.model.score(feature_vectors1)

    # PRINTING THE RESULTS.
    print("Probability for : " + sample.name)
    print(p1)
    probability_vector[i] = p1

# DECIDING THE CLOSEST MATCH.
closest_match = np.argmax(probability_vector)
closest_match_value = np.max(probability_vector)
closest_match_name = (pickle.load(open("gmodel" + str(closest_match + 1), "rb"))).name
print("Testing speech is by "+test_speech_name)
print("Closest Match :")
print(closest_match_name)

# ESTIMATING CONFUSIONS.
# print("Confusion(s):")
# yes_confusion = 0
# for i in range(no_of_speakers):
#     if probability_vector[i] > closest_match_value - 500:
#         if (pickle.load(open("gmodel"+str(i+1), "rb"))).name != closest_match_name:
#             print((pickle.load(open("gmodel"+str(i+1), "rb"))).name)
#             yes_confusion = 1
#
# if yes_confusion == 0:
#     print("--Nil--")
