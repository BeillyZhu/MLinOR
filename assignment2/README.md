## MLinOR
Assignment 2 for the course Machine Learning in OR (FEM21046).

To install all the packages in one go, write 'pip install -r requirements.txt' in the terminal. After installing the packages and using the correct data path, the PSO_BV.py file can already be run without any additional steps to gain the CE and MSE values found in the report. Some further details are provided below.

In all the code random_state = 42 serves as the seed of a Random Number Generator. This ensures that the results are the same for each run, to guarantee reproducibility. The data is randomly split into 80% training set and 20% validation set. 

# PSO_BV.py
Running this file performs the PSO algorithm to train both a regression and classification model, it then uses 5-fold cross validation. For each fold using the validation data, the following is printed: the final value of the fitness function, the classification accuracy, the MSE and the cross-entropy.

It also contains the grid_search function that was used for hyperparameter tuning (should be commented out to avoid the long running time). The parameters in the file are already set to the values found from this grid-search.

# gradientBoosting.py
This file fits a model on the current data using the gradient boosting method developed in the first assignment. As a way to evaluate the performance of the regression model found from PSO, we calculate the MSE using gradient boosting to compare to the MSE from PSO.

# reader.py
This file contains a function that reads the input data. 

# metrics.py
This file contains functions to calculate different measures which could be used in the objective function. This includes for the continuous label the mean squared error and mean absolute error. For the binary label the tested measures are the cross entropy, accuracy, F1-score and area under the ROC-curve.

# diagnostics.py
This file contains some functions to transform the data and get some diagnostics, including: getting the ranges of each feature, splitting the data by binary and continuous feature type, removing features only containing zero values in the data and calculating the feature correlations.


