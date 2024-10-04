import numpy as np

def read(file_path):
    matrix = np.loadtxt(file_path, delimiter=',')
    X = matrix[:,1:]
    y = matrix [:,0]
    return X, y

