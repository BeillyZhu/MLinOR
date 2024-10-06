from reader import read
import pandas as pd
import numpy as np

X, y1, y2 = read("assignment2\Assignment2-Data.csv")
np.set_printoptions(suppress=True)

def get_ranges(X : np.array):
    ranges = []
    if len(X.shape) == 1:
        max_val = np.max(X).item()
        min_val = np.min(X).item()
        ranges.append((min_val,max_val))
    else:
        for col in range(X.shape[1]):
            max_val = np.max(X[:, col]).item()
            min_val = np.min(X[:, col]).item()
            ranges.append((min_val,max_val))
    return ranges

def split_feature_type(X : np.array):
    ranges = get_ranges(X)
    X_c = np.empty((X.shape[0], 1)) #Continuous
    X_b = np.empty((X.shape[0], 1)) #Binary
    for col in range(X.shape[1]):
        if ranges[col][0] == ranges[col][1]:
            continue
        elif ranges[col][0] != 0 or ranges[col][1] != 1:
            X_c = np.column_stack((X_c, X[:, col]))
        else:
            X_b = np.column_stack((X_b, X[:, col]))
    X_c = X_c[:, 1:]
    X_b = X_b[:, 1:]
    return X_c, X_b

def correlation(file_path : str):
    variables = ["y1", "y2"]
    matrix = np.loadtxt(file_path, delimiter=',')
    for i in range(1, matrix.shape[1] - 1):
        variables.append(f"X{i}")
    corr_matrix = np.corrcoef(matrix, rowvar=False)
    corr_df = pd.DataFrame(corr_matrix, index=variables, columns=variables)
    return corr_df

X_c, X_b = split_feature_type(X)
np.savetxt("assignment2\X_continuous.csv", X_c, delimiter=',', fmt='%d')
np.savetxt("assignment2\X_binary.csv", X_b, delimiter=',', fmt='%d')
correlation("assignment2\Assignment2-Data.csv").to_csv("assignment2\correlation.csv")
