"""
Contains Classes that hold information about models.

1. GMMModel: Stores a GMMHMM object from hmmlearn library
2. Model : Stores a MultinomialHMM object and a K-Means Clustered Codebook. (Not used anymore)


"""


class GMMModel(object):  # CLASS FOR GMM - HMM MODEL

    def __init__(self, model_name, name):
        self.model = model_name
        self.name = name


class Model(object):  # CLASS FOR HMM WITH DISCRETE EMISSIONS

    def __init__(self, model_name, codebook, name):
        self.name = name
        self.model = model_name
        self.codebook = codebook  # Object of Type K-Means
