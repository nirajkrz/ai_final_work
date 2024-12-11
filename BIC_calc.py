import math

# Given data for the models
models = [
    {"name": "Model A", "p": 3, "log_likelihood": -12000},
    {"name": "Model B", "p": 6, "log_likelihood": -11500},
    {"name": "Model C", "p": 12, "log_likelihood": -10800},
    {"name": "Model D", "p": 20, "log_likelihood": -9500},
    {"name": "Model E", "p": 35, "log_likelihood": -9400}
]

# Number of data points
n = 10000

# Function to calculate BIC
def calculate_bic(p, log_likelihood, n):
    ln_n = math.log(n)  # Natural logarithm of n
    bic = p * ln_n - 2 * log_likelihood
    return bic

# Calculate BIC for each model and display the result
for model in models:
    bic = calculate_bic(model["p"], model["log_likelihood"], n)
    print(f"{model['name']} BIC: {bic:.2f}")
