import numpy as np
import pandas as pd
from reader import read

file_path = 'Assignment1-Data.csv'
X, y = read(file_path)
n, p = np.shape(X)
print (f"{n} observations with {p} features")
