import numpy as np
import pandas as pd
from reader import read
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# Read the csv file
file_path = 'Assignment1-Data.csv'
X, y = read(file_path)

# Get and report the number of observations and features
n, p = np.shape(X)
print (f"{n} observations with {p} features")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Create a pipeline both for scaling and Elastic Net regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),     # for scaling
    ('elasticnet', ElasticNet(max_iter=5000))     # for applying the elastic net regression
])

# Define the hyperparameter grid
params_grid = {
    'elasticnet__alpha': np.linspace(0.005, 0.01, 10),     #penalty parameter (regularization strength)
    'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]   # mixing parameter (L1 - L2 ratio)
}

# Grid search with 10-fold Cross Validation (CV)
grid_search = GridSearchCV(pipeline, params_grid, cv=10, scoring = 'neg_mean_squared_error')

# Fit the model
grid_search.fit(X_train, y_train)

# Obtain the cross-validation results, evaluate each result and report them
cv_results = grid_search.cv_results_
mean_test_scores = cv_results['mean_test_score']
std_test_scores = cv_results['std_test_score']
params = cv_results['params']

for i in range(len(mean_test_scores)):
    print(f'Fold {i+1}: MSE: {-mean_test_scores[i]:.4f}, (Std: {std_test_scores[i]:.4f}) with parameters {params[i]}')

# Obtain the best model and then test it on the test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Report the best hyperparameters and the MSE
print(f'Best Hyperparameters: {grid_search.best_params_}')
print(f'The Mean Squared Error (MSE) on the test data: {mse}')

print(f'Variance of target: {np.var(y)}')
