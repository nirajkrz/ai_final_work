import itertools

# Define the target sum for a magic square
TARGET_SUM = 15


# Function to calculate the fitness score of a square
def calculate_fitness(square):
    fitness = 0
    # Reshape the 1D square into a 3x3 matrix
    square_2d = [square[i:i + 3] for i in range(0, 9, 3)]

    # Calculate row sums
    row_sums = [sum(row) for row in square_2d]
    fitness += sum(abs(row_sum - TARGET_SUM) for row_sum in row_sums)

    # Calculate column sums
    column_sums = [sum(square_2d[i][j] for i in range(3)) for j in range(3)]
    fitness += sum(abs(column_sum - TARGET_SUM) for column_sum in column_sums)

    # Calculate diagonal sums
    main_diag_sum = sum(square_2d[i][i] for i in range(3))
    anti_diag_sum = sum(square_2d[i][2 - i] for i in range(3))
    fitness += abs(main_diag_sum - TARGET_SUM)
    fitness += abs(anti_diag_sum - TARGET_SUM)

    return fitness


# Function to perform all possible single swaps and find the best fitness
def best_fitness_after_mutation(square):
    best_fitness = float('inf')
    worst_fitness = float('-inf')
    n = len(square)

    # Generate all possible pairs of indices for swaps
    for i, j in itertools.combinations(range(n), 2):
        # Create a copy of the square and swap two elements
        mutated_square = square[:]
        mutated_square[i], mutated_square[j] = mutated_square[j], mutated_square[i]

        # Calculate fitness for the mutated square
        fitness = calculate_fitness(mutated_square)

        # Update the worst fitness if this mutation is worse
        worst_fitness = max(worst_fitness, fitness)

        # Update the best fitness if this mutation is better
        best_fitness = min(best_fitness, fitness)

    return best_fitness, worst_fitness


# Initial valid square
initial_square = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Find the best fitness after one mutation
best_fitness, worst_fitness = best_fitness_after_mutation(initial_square)

print("Best fitness after one mutation:", best_fitness)
print("Worst fitness after one mutation:", worst_fitness)
