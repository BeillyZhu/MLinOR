## MLinOR
Github repository for the course Machine Learning in OR (FEM21046).

To install all the packages in one go, write 'pip install -r requirements.txt' in the terminal. After installing the packages, the elasticNet.py and gradientBoosting.py files can already be run without any additional steps. Some further details are provided below.

In all the code random_state = 42 serves as the seed of a Random Number Generator. This ensures that the results are the same for each run, to guarantee reproducibility. The data is randomly split into 80% training set and 20% test set. 

# reader.py
This file contains a function that reads the input data. The first column of the data is the target variable. The target variable is separated from the feature variables. 

# elasticNet.py
Running this file fits an Elastic Net Regression on the training data. The hyperparameters are tuned using grid search and 10-fold cross validation and the model is tested on the test set, providing an MSE value.

# gradientBoosting.py
Running this file fits a Gradient Boosting model on the training data. The hyperparameters are tuned using randomized search and 10-fold cross validation and the model is tested on the test set, providing an MSE value.