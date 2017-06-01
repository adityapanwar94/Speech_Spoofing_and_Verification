"""
Classifier Program.

Contains the function definition for classifying a given sequence

Once a model is trained, the model parameters are used to classify any given input. Feature Extraction and VQ
is done before passing it here.

When new data is available it is classified by applying Forward Algorithm to all the models and selecting the one with
highest probability. There could be a threshold probability which decides if the input voice is new. If voice is new,
we go to Step 3.
"""

import numpy as np

threshold = 0.000000  # Actual value to be decided.

# all_models : list of models
# observation : a list of numbers which correspond to column index of matrix model.b


def classifier(all_models, observation, t, n):
    probability = np.zeros(len(all_models))
    for model_index in np.arange(len(all_models)):
        probability[model_index] = forward(all_models[model_index], observation, t, n)
    probability[probability < threshold] = 0.0
    # print(probability)
    if np.argmax(probability) > threshold:
        print(np.argmax(probability))

    """
    Optional Steps
    Make a list of probabilities > threshold.
    Choose based on difference.
    """


def forward(model, observation, t, n):
    alpha = np.zeros(t, n)
    alpha[0, :] = np.transpose(np.multiply(model.pi, model.b[:, observation[0]]))  # Initial Value : Return is a Row Vector
    for tindex in np.arange(1, t, 1):
        for j in np.arange(0, n, 1):
            sum1 = 0
            for i in np.arange(0, n, 1):
                sum1 = sum1 + alpha[tindex][i]*model.a[i][j]
            alpha[tindex, j] = np.multiply(model.b[j][observation[tindex]], sum1)
    return np.sum(alpha[t-1, :])

# Normalisation needs to be performedsss
