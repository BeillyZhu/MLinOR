import numpy as np
import pandas as pd
from reader import read
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

file_path = 'Assignment1-Data.csv'
X, y = read(file_path)
n, p = np.shape(X)
print (f"{n} observations with {p} features")

