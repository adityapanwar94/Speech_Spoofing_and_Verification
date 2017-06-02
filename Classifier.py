"""
Classifier Program.

Contains the functions for classifying a given sequence

Once a model is trained, the model parameters are used to classify any given input. Feature Extraction and VQ
is done before passing it here.

When new data is available it is classified by applying Forward Algorithm to all the models and selecting the one with
highest probability. There could be a threshold probability which decides if the input voice is new. If voice is new,
we go to Step 3.
"""

import numpy as np
from sklearn.cluster import KMeans
from Model import Model  # Separate Class that models a HMM

# PARAMETERS
threshold = 0.000000  # Actual value to be decided.

# all_models : list of models
# observation : a list of numbers which correspond to column index of matrix model.b


def quantize(model, feature_vector_list):
    closest_cluster = np.zeros(len(feature_vector_list))
    for feature_vector_index in np.arange(feature_vector_list):
        closest_cluster[feature_vector_index] = model.codebook.predict(feature_vector_list[feature_vector_index])
    return closest_cluster


def forward(observation_list, model, n):
    alpha = np.zeros((len(observation_list), n))
    for i in np.arange(0, n, 1):
        alpha[0][i] = model.pi[i] * model.b[i][0]
    for t in np.arange(0, len(observation_list), 1):
        for j in np.arange(0, n, 1):
            for i in np.arange(0, n, 1):
                alpha[t+1][j] = alpha[t+1][j] + alpha[t][i] * model.a[i][j] * model.b[j][t+1]
    return np.sum(alpha[t-1, :])
# Normalisation needs to be performed.


def classifier(all_models, mfcc_features, n):
    probability = np.zeros(len(all_models))
    t = len(mfcc_features)
    for model_index in np.arange(len(all_models)):
        observations = quantize(all_models[model_index], mfcc_features)
        probability[model_index] = forward(all_models[model_index], observations, t, n)
    probability[probability < threshold] = 0.0
    # print(probability)
    if np.argmax(probability) > threshold:
        print(np.argmax(probability))

"""
Optional Steps
Make a list of probabilities > threshold.
Choose based on difference.
"""