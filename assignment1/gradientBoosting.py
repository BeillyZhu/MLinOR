import numpy as np
import pandas as pd
from reader import read
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

# Read CSV file
data = np.loadtxt('Assignment2-Data.csv', delimiter=',')
X = data[:, 2:]
y = data[:, 1]  # y2 for regression



# Split into training sets and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning
param_distributions = {
   'n_estimators': [100, 170, 180, 200, 250],   # Number of trees to build
    'learning_rate': [0.1, 0.09, 0.085, 0.01],  # Each tree contribution(usually between 0.01 and 0.1)
    'max_depth': [4, 5, 6],                     # Maximum tree depth
    'min_samples_split': [5, 6],                # Minimum of samples to split a leaf (avoid overfitting)
    'min_samples_leaf': [3, 4, 5],              # Minimum of samples in each leaf (avoid overfitting)
    'subsample': [1.0, 0.8, 0.6],               # Proportion of training data to build each tree, when < 1 a random subset of training data is chosen for each tree
}

gbr = GradientBoostingRegressor(random_state=42)

random_search = RandomizedSearchCV( # Lower running time than grid search
    estimator=gbr,
    param_distributions=param_distributions,
    scoring='neg_mean_squared_error',
    cv=10,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Model with best hyperparameters
best_gbr = random_search.best_estimator_

# Predict on the test data
y_pred = best_gbr.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
for i in range(len(y_test)):
    print(f"the actual value {y_test[i]} vs the prediction {y_pred[i]}")


