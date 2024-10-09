import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler

# L1 regularization
# 10-fold CV
# Standardizing the features, normalizing the classification loss and regression loss in the objective
# Standardize y_regression values
# Comparision with the actual values in the end 

# Load Dataset
data = np.loadtxt('Assignment2-Data.csv', delimiter=',')
X = data[:, 2:]
y_classification = data[:, 0]  # y1 for classification
y_regression = data[:, 1]  # y2 for regression

# Standardize the features and the regression target
scaler = StandardScaler()
X = scaler.fit_transform(X)
scaler_y = StandardScaler()
y_regression = scaler_y.fit_transform(y_regression.reshape(-1, 1)).flatten()  # Standardize y_regression

# PSO Parameters
POPULATION_SIZE = 50  # Number of particles in the swarm
MAX_ITERATIONS = 200  # Maximum number of iterations for the optimization process
INERTIA_WEIGHT = 0.8  # Inertia weight to control exploration and exploitation
COGNITIVE_COEFF = 2.0  # Cognitive coefficient to control the influence of personal best position
SOCIAL_COEFF = 1.0  # Social coefficient to control the influence of the global best position
ALPHA = 0.7  # Weight parameter to balance classification and regression tasks
LAMBDA = 0.2  # L2 Regularization strength

# Initialize population (each particle is a set of weights for classification and regression)
def initialize_particles(pop_size, feature_count):
    # Each particle has weights for classification and regression, plus bias terms
    return [np.random.uniform(-1, 1, 2 * feature_count + 2) for _ in range(pop_size)], [np.random.uniform(-1, 1, 2 * feature_count + 2) for _ in range(pop_size)]

# Sigmoid function for classification
def sigmoid(x):
    x = np.clip(x, -1000, 1000)  # Clip to prevent overflow
    return 1 / (1 + np.exp(-x))

# Fitness Function
def evaluate_fitness(particle, X, y_class, y_reg):
    feature_count = X.shape[1]
    W_class = particle[:feature_count]  # Weights for classification task
    W_reg = particle[feature_count:2*feature_count]  # Weights for regression task
    bias_class = particle[-2]  # Bias term for classification
    bias_reg = particle[-1]  # Bias term for regression

    # Classification predictions
    y_class_pred_prob = sigmoid(np.dot(X, W_class) + bias_class)
    y_class_pred_prob = np.clip(y_class_pred_prob, 1e-15, 1 - 1e-15)  # Clip probabilities to avoid log loss errors
    classification_loss = log_loss(y_class, y_class_pred_prob)
    # Normalize classification loss
    classification_loss /= len(y_class)

    # Regression predictions
    y_reg_pred = np.dot(X, W_reg) + bias_reg
    regression_mse = mean_squared_error(y_reg, y_reg_pred)
    # Normalize regression loss
    regression_mse /= len(y_reg)

    # L2 Regularization penalty
    l2_penalty = LAMBDA * (np.sum(W_class**2) + np.sum(W_reg**2))

    # Combined fitness (lower is better for both log loss and MSE)
    fitness = ALPHA * (1 / (1 + classification_loss)) + (1 - ALPHA) * (1 / (1 + regression_mse)) - l2_penalty
    return fitness

# 10-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold = 1
for train_index, val_index in kf.split(X):
    print(f"Fold {fold}")
    fold += 1

    X_train, X_val = X[train_index], X[val_index]
    y_class_train, y_class_val = y_classification[train_index], y_classification[val_index]
    y_reg_train, y_reg_val = y_regression[train_index], y_regression[val_index]

    # PSO Algorithm
    # Initialize particles and velocities
    particles, velocities = initialize_particles(POPULATION_SIZE, X_train.shape[1])
    # Initialize personal best positions and fitnesses for each particle
    personal_best_positions = particles.copy()
    personal_best_fitnesses = [evaluate_fitness(p, X_train, y_class_train, y_reg_train) for p in particles]
    # Initialize global best position and fitness
    global_best_position = personal_best_positions[np.argmax(personal_best_fitnesses)]
    global_best_fitness = max(personal_best_fitnesses)

    # PSO main loop
    for iteration in range(MAX_ITERATIONS):
        for i in range(POPULATION_SIZE):
            # Update velocity for each particle
            r1, r2 = np.random.rand(), np.random.rand()  # Random factors to add stochasticity
            velocities[i] = (INERTIA_WEIGHT * velocities[i] +
                             COGNITIVE_COEFF * r1 * (personal_best_positions[i] - particles[i]) +
                             SOCIAL_COEFF * r2 * (global_best_position - particles[i]))
            # Update position of each particle
            particles[i] = particles[i] + velocities[i]

            # Evaluate the fitness of the updated particle
            current_fitness = evaluate_fitness(particles[i], X_train, y_class_train, y_reg_train)

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
        print(f"Iteration {iteration + 1}, Best Fitness: {global_best_fitness}")

    # Evaluate final solution on validation data
    final_fitness = evaluate_fitness(global_best_position, X_val, y_class_val, y_reg_val)
    print(f"Final Validation Fitness: {final_fitness}")

    # Print predictions for a random sample of validation data
    sample_size = min(10, len(X_val))  # Choose a sample size of 10 or less if validation set is smaller
    sample_indices = np.random.choice(len(X_val), sample_size, replace=False)
    X_sample = X_val[sample_indices]
    y_class_sample_actual = y_class_val[sample_indices]
    y_reg_sample_actual = scaler_y.inverse_transform(y_reg_val[sample_indices].reshape(-1, 1)).flatten()  # Inverse transform

    # Predictions using the global best position
    feature_count = X_sample.shape[1]
    W_class = global_best_position[:feature_count]  # Weights for classification task
    W_reg = global_best_position[feature_count:2*feature_count]  # Weights for regression task
    bias_class = global_best_position[-2]  # Bias term for classification
    bias_reg = global_best_position[-1]  # Bias term for regression

    # Classification and regression predictions for the sample
    y_class_pred_sample = sigmoid(np.dot(X_sample, W_class) + bias_class) >= 0.5  # Threshold at 0.5 for classification
    y_reg_pred_sample_standardized = np.dot(X_sample, W_reg) + bias_reg
    y_reg_pred_sample = scaler_y.inverse_transform(y_reg_pred_sample_standardized.reshape(-1, 1)).flatten()  # Inverse transform

    # Print actual vs predicted values
    print("\nSample Predictions vs Actual Values:")
    for i in range(sample_size):
        print(f"Sample {i + 1}:")
        print(f"  Classification - Actual: {y_class_sample_actual[i]}, Predicted: {y_class_pred_sample[i]}")
        print(f"  Regression - Actual: {y_reg_sample_actual[i]:.3f}, Predicted: {y_reg_pred_sample[i]:.3f}")