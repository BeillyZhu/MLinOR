import numpy as np

def read(file_path):
    matrix = np.loadtxt(file_path, delimiter=',')
    X = matrix[:,2:]
    y1 = matrix[:,0]
    y2 = matrix[:,1]
    return X, y1, y2

