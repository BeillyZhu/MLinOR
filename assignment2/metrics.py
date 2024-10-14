from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, mean_absolute_error, fbeta_score, roc_auc_score
from diagnostics import *
from sklearn.preprocessing import StandardScaler

# All functions are fit functions. This means the higher the value, the better.



# ------Continuous------
# y_true should be the continuous label, y_pred should be the predicted value

# Mean Squared Error
def inverse_MSE(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred)
    return (1 / (1 + MSE))

# Mean Absolute Error
def inverse_MAE(y_true, y_pred):
    MAE = mean_absolute_error(y_true, y_pred)
    return (1/ (1+MAE))




# ------Binary------
# y_true should be the binary label, predicted_prob should be the predicted probability

# Mean Cross Entropy
def inverse_CE(y_true, predicted_prob):
    CE = log_loss(y_true, predicted_prob)
    return (1/(1 + CE))

# Accuracy
def accuracy(y_true, predicted_prob):
    y_pred = predicted_prob > 0.5
    return accuracy_score(y_true, y_pred)

# F1-score, balance between recall and precision
def F1_score(y_true, predicted_prob):
    y_pred = predicted_prob > 0.5
    return fbeta_score(y_true, y_pred, beta=1)

# Area Under the ROC-curve, which plots the True Positive Rate and the False Positive Rate
def ROC_area(y_true, predicted_prob):
    return roc_auc_score(y_true, predicted_prob)
    

# ------Universal Test Measure------
def test_measure(ALPHA, y_reg, y_reg_pred, y_class, predicted_prob):
    return ALPHA * inverse_CE(y_class, predicted_prob) + (1 - ALPHA) * inverse_MSE(y_reg, y_reg_pred)

