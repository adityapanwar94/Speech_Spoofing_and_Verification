"""
Contains Classes that hold information about models.

"""


class GMMModel(object):  # CLASS FOR GMM - HMM MODEL

    def __init__(self, model_name, name):
        self.model = model_name
        self.name = name


class Model(object):  # CLASS FOR HMM WITH DISCRETE EMISSIONS

    def __init__(self, n, a, b, pi, codebook, n_clusters, name):
        self.name = name
        self.n = n
        self.A = a
        self.B = b
        self.pi = pi
        self.codebook = codebook  # Object of Type K-Means


