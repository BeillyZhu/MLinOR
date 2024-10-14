import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from diagnostics import *
from metrics import *


# Load Dataset
data = np.loadtxt('Assignment2-Data.csv', delimiter=',')
X = data[:, 2:]  # Features starting from the third column
y_classification = data[:, 0]  # y1 for classification task (first column)
y_regression = data[:, 1]  # y2 for regression task (second column)

# Split data into training (80%) and testing (20%)
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_classification, y_regression, test_size=0.2, random_state=42
)

# Standardize the features and the regression target
scaler = StandardScaler()  # Standard scaler to normalize feature values
X_train = scaler.fit_transform(X_train)  # Standardize training features
X_test = scaler.transform(X_test)  # Standardize test features
scaler_y = StandardScaler()  # Standard scaler for regression target
y_reg_train = scaler_y.fit_transform(y_reg_train.reshape(-1, 1)).flatten()  # Standardize y_regression for training

# PSO Parameters
POPULATION_SIZE = 50  # Number of particles in the swarm
MAX_ITERATIONS = 100  # Maximum number of iterations for the optimization process
INERTIA_WEIGHT = 0.8  # Inertia weight to control exploration and exploitation
COGNITIVE_COEFF = 1.75  # Cognitive coefficient to control the influence of personal best position
SOCIAL_COEFF = 0.65  # Social coefficient to control the influence of the global best position
ALPHA = 0.5  # Weight parameter to balance classification and regression tasks
LAMBDA = 0.01 # Regularization strength


# Initialize population (each particle is a set of weights for classification and regression)
def initialize_particles(pop_size, feature_count):
    # Each particle has weights for classification and regression, plus bias terms
    # The total number of elements per particle is (2 * feature_count + 2):
    #   - feature_count weights for classification
    #   - feature_count weights for regression
    #   - 1 bias for classification
    #   - 1 bias for regression
    return [np.random.RandomState(42).uniform(-1, 1, 2 * feature_count + 2) for _ in range(pop_size)], [np.random.RandomState(42).uniform(-1, 1, 2 * feature_count + 2) for _ in range(pop_size)]

# Sigmoid function for classification
def sigmoid(x):
    # Clip the input values to prevent overflow when applying the exponential function
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Fitness Function
def evaluate_fitness(particle, X, y_class, y_reg, class_fit_func, reg_fit_func, penalty_dimension):
    feature_count = X.shape[1]  # Number of features in the dataset
    W_class = particle[:feature_count]  # Weights for classification task
    W_reg = particle[feature_count:2*feature_count]  # Weights for regression task
    bias_class = particle[-2]  # Bias term for classification
    bias_reg = particle[-1]  # Bias term for regression

    # Classification predictions
    predicted_prob = sigmoid(np.dot(X, W_class) + bias_class)
    classification_fit = class_fit_func(y_class, predicted_prob)  # Compute loss for classification


    # Regression predictions
    y_reg_pred = np.dot(X, W_reg) + bias_reg  # Predict regression values
    regression_fit = reg_fit_func(y_reg, y_reg_pred)

    

    if penalty_dimension <= 0:
        penalty = 0
    else:
        penalty = LAMBDA  * (np.sum(W_class**penalty_dimension) + np.sum(W_reg**penalty_dimension))

    # Combined fitness
    # The fitness function balances classification accuracy and regression MSE
    fitness = ALPHA * classification_fit + (1 - ALPHA) * regression_fit - penalty
    return fitness



# 5-Fold Cross-Validation on Training Set
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Initialize 5-fold cross-validation
fold = 1  # Initialize fold counter
all_samples = []  # Store samples across all folds
all_fitness = []  # Store fitness across all folds
all_class_accuracies = []  # Store classification accuracies across all folds
all_regression_mses = []  # Store regression MSEs across all folds
for train_index, val_index in kf.split(X_train):
    print(f"Fold {fold}")  # Print current fold number
    fold += 1

    # Split data into training and validation sets for the current fold
    X_fold_train, X_val = X_train[train_index], X_train[val_index]
    y_class_fold_train, y_class_val = y_class_train[train_index], y_class_train[val_index]
    y_reg_fold_train, y_reg_val = y_reg_train[train_index], y_reg_train[val_index]

    # PSO Algorithm
    # Initialize particles and velocities
    particles, velocities = initialize_particles(POPULATION_SIZE, X_fold_train.shape[1])
    # Initialize personal best positions and fitnesses for each particle
    personal_best_positions = particles.copy()  # Each particle starts with its initial position as its personal best
    personal_best_fitnesses = [evaluate_fitness(p, X_fold_train, y_class_fold_train, y_reg_fold_train, inverse_MCE, inverse_MSE, 2) for p in particles]  # Evaluate fitness for each particle
    # Initialize global best position and fitness
    global_best_position = personal_best_positions[np.argmax(personal_best_fitnesses)]  # Find the particle with the best fitness
    global_best_fitness = max(personal_best_fitnesses)  # Set the best fitness value

    # PSO main loop
    for iteration in range(MAX_ITERATIONS):
        for i in range(POPULATION_SIZE):
            # Update velocity for each particle
            r1, r2 = np.random.RandomState(42).rand(), np.random.RandomState(42).rand()  # Random factors to add stochasticity
            velocities[i] = (INERTIA_WEIGHT * velocities[i] +
                             COGNITIVE_COEFF * r1 * (personal_best_positions[i] - particles[i]) +
                             SOCIAL_COEFF * r2 * (global_best_position - particles[i]))
            # Update position of each particle
            particles[i] = particles[i] + velocities[i]

            # Evaluate the fitness of the updated particle
            current_fitness = evaluate_fitness(particles[i], X_fold_train, y_class_fold_train, y_reg_fold_train, inverse_MCE, inverse_MSE, 2)

            # Update personal best if current fitness is better
            if current_fitness > personal_best_fitnesses[i]:
                personal_best_positions[i] = particles[i]
                personal_best_fitnesses[i] = current_fitness

        # Update global best if a better personal best is found
        best_particle_index = np.argmax(personal_best_fitnesses)
        if personal_best_fitnesses[best_particle_index] > global_best_fitness:
            global_best_position = personal_best_positions[best_particle_index]
            global_best_fitness = personal_best_fitnesses[best_particle_index]

        # Print best fitness of the current iteration
        # print(f"Iteration {iteration + 1}, Best Fitness: {global_best_fitness}")

    # Evaluate final solution on validation data
    final_fitness = evaluate_fitness(global_best_position, X_val, y_class_val, y_reg_val, inverse_MCE, inverse_MSE, 2)
    all_fitness.append(final_fitness)

    # Calculate classification accuracy and regression MSE on validation set
    W_class = global_best_position[:X_val.shape[1]]  # Weights for classification task
    W_reg = global_best_position[X_val.shape[1]:2*X_val.shape[1]]  # Weights for regression task
    bias_class = global_best_position[-2]  # Bias term for classification
    bias_reg = global_best_position[-1]  # Bias term for regression

    # Classification predictions on validation set
    y_class_pred_prob_val = sigmoid(np.dot(X_val, W_class) + bias_class)
    y_class_pred_val = (y_class_pred_prob_val >= 0.5).astype(int)  # Threshold at 0.5 for classification
    classification_accuracy = accuracy_score(y_class_val, y_class_pred_val)
    classification_mce = log_loss(y_class_val, y_class_pred_prob_val)/ len(y_class_val)
    all_class_accuracies.append(classification_accuracy)

    # Regression predictions on validation set
    y_reg_pred_val_standardized = np.dot(X_val, W_reg) + bias_reg
    y_reg_pred_val = scaler_y.inverse_transform(y_reg_pred_val_standardized.reshape(-1, 1)).flatten()  # Inverse transform to original scale
    regression_mse_val = mean_squared_error(scaler_y.inverse_transform(y_reg_val.reshape(-1, 1)).flatten(), y_reg_pred_val)
    all_regression_mses.append(regression_mse_val)

    # Store predictions for a random sample of validation data
    sample_size = min(10, len(X_val))  # Choose a sample size of 10 or less if validation set is smaller
    sample_indices = np.random.choice(len(X_val), sample_size, replace=False)  # Randomly select sample indices
    X_sample = X_val[sample_indices]
    y_class_sample_actual = y_class_val[sample_indices]  # Actual classification labels for the sample
    y_reg_sample_actual = scaler_y.inverse_transform(y_reg_val[sample_indices].reshape(-1, 1)).flatten()  # Inverse transform to original scale

    # Predictions using the global best position
    feature_count = X_sample.shape[1]
    W_class = global_best_position[:feature_count]  # Weights for classification task
    W_reg = global_best_position[feature_count:2*feature_count]  # Weights for regression task
    bias_class = global_best_position[-2]  # Bias term for classification
    bias_reg = global_best_position[-1]  # Bias term for regression

    # Classification and regression predictions for the sample
    y_class_pred_sample = sigmoid(np.dot(X_sample, W_class) + bias_class) >= 0.5  # Threshold at 0.5 for classification
    y_reg_pred_sample_standardized = np.dot(X_sample, W_reg) + bias_reg  # Predict regression values (standardized)
    y_reg_pred_sample = scaler_y.inverse_transform(y_reg_pred_sample_standardized.reshape(-1, 1)).flatten()  # Inverse transform to original scale

    # Store sample predictions and actual values
    for i in range(sample_size):
        all_samples.append((y_class_sample_actual[i], y_class_pred_sample[i], y_reg_sample_actual[i], y_reg_pred_sample[i]))

    # Report metrics for the fold
    print(f"Fold {fold - 1} Summary:")
    print(f"  Final Fitness: {final_fitness}")
    print(f"  Classification Accuracy: {classification_accuracy:.4f}")
    print(f"  Regression MSE: {regression_mse_val:.4f}")
    print(f"  Classification MCE {classification_mce:.8f}")

# Print predictions for a total of 20 samples across all folds
# print("\nSample Predictions vs Actual Values:")
# for i, (y_class_actual, y_class_pred, y_reg_actual, y_reg_pred) in enumerate(random.sample(all_samples, min(20, len(all_samples)))):
    # print(f"Sample {i + 1}:")
    # print(f"  Classification - Actual: {y_class_actual}, Predicted: {y_class_pred}")
    # print(f"  Regression - Actual: {y_reg_actual:.3f}, Predicted: {y_reg_pred:.3f}")


# Grid Search Setup for Hyperparameter Tuning
# Define parameter search space
cognitive_coeff_range = np.linspace(0.5, 3.0, 5)  # Expanded range for cognitive coefficient with smaller step size
social_coeff_range = np.linspace(0.2, 2.0, 5)  # Expanded range for social coefficient with smaller step size
lambda_range = np.linspace(0.01, 0.5, 5)  
class_fit_range = [inverse_MCE]
                #    , accuracy, F1_score, ROC_area]
reg_fit_range = [inverse_MSE]
                #  , inverse_MAE]

def grid_search():
    best_fitness = -np.inf  # We are maximizing fitness
    best_params = None
    
    # Initialize KFold for cross-validation (reuse the same kf from earlier code)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Iterate over all possible combinations of hyperparameters
    for cognitive_coeff in cognitive_coeff_range:
        for social_coeff in social_coeff_range:
            for lambda_param in lambda_range:
                for class_fit in class_fit_range:
                    for reg_fit in reg_fit_range:
                        # Update global hyperparameters
                        global COGNITIVE_COEFF, SOCIAL_COEFF, LAMBDA
                        COGNITIVE_COEFF = cognitive_coeff
                        SOCIAL_COEFF = social_coeff
                        LAMBDA = lambda_param

                        fold_fitnesses = []  # List to store fitness for each fold

                        # Perform K-fold cross-validation
                        for train_index, val_index in kf.split(X_train):
                            # Split data into training and validation sets for current fold
                            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
                            y_class_fold_train, y_class_fold_val = y_class_train[train_index], y_class_train[val_index]
                            y_reg_fold_train, y_reg_fold_val = y_reg_train[train_index], y_reg_train[val_index]
                            
                            # Re-initialize particles and run PSO on this fold
                            particles, velocities = initialize_particles(POPULATION_SIZE, X_fold_train.shape[1])
                            personal_best_positions = particles.copy()
                            personal_best_fitnesses = [evaluate_fitness(p, X_fold_train, y_class_fold_train, y_reg_fold_train, class_fit, reg_fit, 2) for p in particles]
                            global_best_position = personal_best_positions[np.argmax(personal_best_fitnesses)]
                            global_best_fitness = max(personal_best_fitnesses)
                            
                            # PSO Optimization loop for the current fold
                            for iteration in range(MAX_ITERATIONS):
                                for i in range(POPULATION_SIZE):
                                    # Update velocity and position of each particle
                                    r1, r2 = np.random.RandomState(42).rand(), np.random.RandomState(42).rand()
                                    velocities[i] = (INERTIA_WEIGHT * velocities[i] +
                                                    COGNITIVE_COEFF * r1 * (personal_best_positions[i] - particles[i]) +
                                                    SOCIAL_COEFF * r2 * (global_best_position - particles[i]))
                                    particles[i] = particles[i] + velocities[i]
                                    
                                    # Evaluate updated fitness
                                    current_fitness = evaluate_fitness(particles[i], X_fold_train, y_class_fold_train, y_reg_fold_train, class_fit, reg_fit, 2)
                                    
                                    # Update personal best if fitness is improved
                                    if current_fitness > personal_best_fitnesses[i]:
                                        personal_best_positions[i] = particles[i]
                                        personal_best_fitnesses[i] = current_fitness
                                
                                # Update global best if fitness is improved
                                best_particle_index = np.argmax(personal_best_fitnesses)
                                if personal_best_fitnesses[best_particle_index] > global_best_fitness:
                                    global_best_position = personal_best_positions[best_particle_index]
                                    global_best_fitness = personal_best_fitnesses[best_particle_index]

                            # After PSO optimization, evaluate final fitness on validation set. The common fitness measure uses cross entropy and MSE.
                            final_fitness = evaluate_fitness(global_best_position, X_fold_val, y_class_fold_val, y_reg_fold_val, inverse_MCE, inverse_MSE, 2)
                            fold_fitnesses.append(final_fitness)

                        # Calculate the average fitness across all folds
                        avg_fitness = np.mean(fold_fitnesses)

                        # If average fitness is better, update best hyperparameters
                        if avg_fitness > best_fitness:
                            best_fitness = avg_fitness
                            best_params = (cognitive_coeff, social_coeff, lambda_param, class_fit, reg_fit)
                            print("Found new best")
                        print(f"COGNITIVE_COEFF: {cognitive_coeff}, SOCIAL_COEFF: {social_coeff}, LAMBDA: {lambda_param}, CLASS_FIT: {class_fit.__name__}, REG_FIT: {reg_fit.__name__}")
                        print(f"Fitness: {avg_fitness}")

    return best_params, best_fitness

# Run grid search
# best_params, best_fitness = grid_search()
# print(f"Grid Search Best Hyperparameters - COGNITIVE_COEFF: {best_params[0]}, SOCIAL_COEFF: {best_params[1]}, LAMBDA: {best_params[2]}, CLASS_FIT: {best_params[3].__name__}, REG_FIT: {best_params[4].__name__}")
# print(f"Best Fitness from Grid Search: {best_fitness}")
