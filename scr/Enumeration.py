# -----------import-library----------------
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import csv
import math
from numba import cuda

def enum (matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
     
    """ 
    GPU computing
    """
    n = matrix.shape[1]
    
    matrix_enum: np.ndarray
    matrix_out: np.ndarray
    
    matrix_enum = np.zeros([n*n,n])
    matrix_out = np.zeros(n*n)

    for i in range(n):
        for j in range(n):
            for z in range(n):
                matrix_enum[i*n+j,z] = matrix[i,j] * matrix[i,z]

    for i in range(n*n):
        for j in range(n):
            matrix_out[i]=matrix_out[i] + matrix_enum[i,j] * weights[j]
    return matrix_out
