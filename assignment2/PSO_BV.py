import numpy as np
import random
from sklearn.metrics import accuracy_score,log_loss, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
data = np.loadtxt('Assignment2-Data.csv', delimiter=',')
X = data[:, 2:]
y_classification = data[:, 0]  # y1 for classification
y_regression = data[:, 1]  # y2 for regression

# Split data into training and testing sets
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_classification, y_regression, test_size=0.2, random_state=42
)

# Standardize the features and the regression
scaler = StandardScaler()
X = scaler.fit_transform(X)
scaler_y = StandardScaler()
y_regression = scaler_y.fit_transform(y_regression.reshape(-1, 1)).flatten()  # Standardize y_regression

# PSO Parameters
POPULATION_SIZE = 170  # Number of particles in the swarm
MAX_ITERATIONS = 100  # Maximum number of iterations for the optimization process
INERTIA_WEIGHT = 0.9  # Inertia weight to control exploration and exploitation
COGNITIVE_COEFF = 1  # Cognitive coefficient to control the influence of personal best position (with 0.5 takes longer)
SOCIAL_COEFF = 2  # Social coefficient to control the influence of the global best position (with 2.5 takes longer)
ALPHA = 0.5  # Weight parameter to balance classification and regression tasks

# Initialize population (each particle is a set of weights for classification and regression)
def initialize_particles(pop_size, feature_count):
    # Each particle has weights for classification and regression, plus bias terms
    return [np.random.uniform(-1, 1, 2 * feature_count + 2) for _ in range(pop_size)], [np.random.uniform(-1, 1, 2 * feature_count + 2) for _ in range(pop_size)]

# Sigmoid function for classification
def sigmoid(x):
    x = np.clip(x, -500, 500)  # Clip values to prevent overflow
    return 1 / (1 + np.exp(-x))


# Fitness Function
def evaluate_fitness(particle, X, y_class, y_reg):
    feature_count = X.shape[1]
    W_class = particle[:feature_count]  # Weights for classification task
    W_reg = particle[feature_count:2*feature_count]  # Weights for regression task
    bias_class = particle[-2]  # Bias term for classification
    bias_reg = particle[-1]  # Bias term for regression

    # Classification predictions
    predicted_prob = sigmoid(np.dot(X, W_class) + bias_class)
    classification_loss = log_loss(y_class, predicted_prob) / X.shape[0]

    # Regression predictions
    y_reg_pred = np.dot(X, W_reg) + bias_reg
    regression_mse = mean_squared_error(y_reg, y_reg_pred)

    # Combined fitness
    # The fitness function balances classification accuracy and regression MSE
    fitness = ALPHA * classification_loss + (1 - ALPHA) * (1 / (1 + regression_mse))
    return fitness

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

    # Display regression and classification predictions vs real values
    
# Evaluate final solution on test data
final_fitness = evaluate_fitness(global_best_position, X_test, y_class_test, y_reg_test)
print(f"Final Test Fitness: {final_fitness}")
