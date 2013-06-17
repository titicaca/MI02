import numpy as np
from numpy import *
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def get_C(data, dimension):
    C = [[0 for i in range(dimension)] for j in range(dimension)]
    p = len(data[0])
    m = [0 for i in range(dimension)]
    for i in range(dimension):
        m[i] = sum(data[i])/p
    for i in range(dimension):
        for j in range(dimension):
            for a in range(p):
                C[i][j] += ((data[i][a] - m[i]) * (data[j][a] - m[j]))/p
    return C
    

def get_PC(data, dimension, nume):
    C = get_C(data, dimension)
    if nume == dimension:
        evals, evecs = np.linalg.eig(asmatrix(C))
    else:
        evals, evecs = sp.sparse.linalg.eigs(asmatrix(C), k = nume)
    return evals, evecs
