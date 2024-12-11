import numpy as np
import matplotlib.pyplot as plt

def compute_dtw(signal1, signal2):
    n = len(signal1)
    m = len(signal2)

    # Initialize the DTW table with infinity
    dtw_table = np.full((n + 1, m + 1), float('inf'))
    dtw_table[0, 0] = 0

    # Fill the DTW table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(signal1[i - 1] - signal2[j - 1])
            dtw_table[i, j] = cost + min(dtw_table[i - 1, j],    # Insertion
                                         dtw_table[i, j - 1],    # Deletion
                                         dtw_table[i - 1, j - 1]) # Match

    return dtw_table

def find_min_cost_path(dtw_table):
    # Start from the bottom-right corner of the table
    i, j = len(dtw_table) - 1, len(dtw_table[0]) - 1
    path = [(i - 1, j - 1)]

    while i > 1 or j > 1:
        # Get the costs of the three possible moves
        diag_cost = dtw_table[i - 1, j - 1] if i > 1 and j > 1 else float('inf')
        left_cost = dtw_table[i, j - 1] if j > 1 else float('inf')
        down_cost = dtw_table[i - 1, j] if i > 1 else float('inf')

        # Choose the direction with the minimum cost, prioritize diagonal in case of a tie
        if diag_cost <= left_cost and diag_cost <= down_cost:
            i, j = i - 1, j - 1  # Move diagonally
        elif left_cost < down_cost:
            j -= 1  # Move left
        else:
            i -= 1  # Move down

        path.append((i - 1, j - 1))

    path.reverse()
    return path

# Calculate the L-2 Distance based on the given formula
def calculate_distance(path, signal1, signal2):
    distance_squared = 0
    for i in range(len(path) - 1):
        vi = abs(signal1[path[i][0]] - signal2[path[i][1]])
        vi_next = abs(signal1[path[i + 1][0]] - signal2[path[i + 1][1]])
        distance_squared += (vi - vi_next) ** 2

    return distance_squared



def plot_dtw_table(dtw_table, signal1, signal2, path):
    # Plot the DTW table with labels, grid, and min-cost path
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display the DTW table
    cax = ax.matshow(dtw_table[1:, 1:], cmap='viridis')
    fig.colorbar(cax)

    # Set axis labels with signal values
    ax.set_xticks(np.arange(len(signal2)))
    ax.set_yticks(np.arange(len(signal1)))
    ax.set_xticklabels(signal2)
    ax.set_yticklabels(signal1)

    # Set labels and title
    ax.set_xlabel("Signal 2 Values")
    ax.set_ylabel("Signal 1 Values")
    ax.set_title("DTW Cost Matrix with Min-Cost Path (Diagonal Priority)")

    # Show grid
    ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)

    # Annotate the DTW table with values
    for i in range(len(signal1)):
        for j in range(len(signal2)):
            ax.text(j, i, f"{dtw_table[i + 1, j + 1]:.1f}",
                    ha='center', va='center', color='white')

    # Plot the min-cost path
    path_x = [j for _, j in path]
    path_y = [i for i, _ in path]
    ax.plot(path_x, path_y, color='red', linewidth=2, marker='o', markersize=5)

    plt.show()




# Example usage
signal1 = [3, 2, 0, 1,4,5, 6,7,2,2,1]
signal2 = [4, 2, 1, 0, 5, 5,7,7,3,2,1]

signal3 = [5, 3,1,7,8,6,9,8,6,3,2]
signal4 = [3,2,0,1,4,5,6,7,2,2,1]

A = [4, 6, 8, 8, 5, 4, 3, 7]
B = [3, 4, 6, 9, 8, 5, 2, 6]

songa = [21, 23, 22, 25, 26, 22, 19, 16]
songb = [22, 20, 17, 15, 18, 20, 15, 17]
songc = [23, 24, 25, 27, 26, 22, 20, 18]

dtw_table = compute_dtw(A, B)
min_cost_path = find_min_cost_path(dtw_table)


dist = calculate_distance(min_cost_path, A, B)

print("distance between signals", dist)
print("DTW Table:")
print(dtw_table)
print("Min-Cost Path:", min_cost_path)

# Plot the DTW table with the min-cost path
plot_dtw_table(dtw_table, songb, songc, min_cost_path)


