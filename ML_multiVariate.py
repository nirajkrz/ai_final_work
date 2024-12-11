import numpy as np
import math


# Function to calculate the multivariate normal distribution PDF for 2D Gaussian
def multivariate_normal(x, mu, sigma):
    # Calculate the determinant of Sigma
    det_sigma = np.linalg.det(sigma)

    # Calculate the inverse of Sigma
    inv_sigma = np.linalg.inv(sigma)

    # Calculate (x - mu)^T * Sigma^-1 * (x - mu)
    diff = x - mu
    exponent = -0.5 * np.dot(np.dot(diff.T, inv_sigma), diff)

    # Compute the PDF
    pdf = (1 / (2 * math.pi * np.sqrt(det_sigma))) * np.exp(exponent)

    return pdf


# Given data
x_n = np.array([15, 40])  # Data point (X = 15, Y = 40)

# Parameters for Component 1
mu_1 = np.array([10, 30])
sigma_1 = np.array([[5, 0], [0, 5]])

# Parameters for Component 2
mu_2 = np.array([18, 45])
sigma_2 = np.array([[8, 0], [0, 8]])

# Mixture weights
pi_1 = 0.5
pi_2 = 0.5

# Calculate N(x_n | mu_1, sigma_1) for Component 1
N_1 = multivariate_normal(x_n, mu_1, sigma_1)

# Calculate N(x_n | mu_2, sigma_2) for Component 2
N_2 = multivariate_normal(x_n, mu_2, sigma_2)

# Responsibility of Component 1 (gamma_1)
gamma_1 = (pi_1 * N_1) / (pi_1 * N_1 + pi_2 * N_2)

# Responsibility of Component 2 (gamma_2)
gamma_2 = (pi_2 * N_2) / (pi_1 * N_1 + pi_2 * N_2)

# Print the responsibility rounded to 8 decimal places
print(f"Responsibility of Component 2 (γ2) for data point (15, 40): {gamma_2:.8f}")

# Print the responsibility rounded to 8 decimal places
print(f"Responsibility of Component 1 (γ1) for data point (15, 40): {gamma_1:.8f}")
