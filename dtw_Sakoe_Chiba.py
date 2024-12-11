import numpy as np
import matplotlib.pyplot as plt


def dtw_with_bound_margin1(signal1, signal2, window_size=1):
    n = len(signal1)
    m = len(signal2)

    # Initialize the DTW matrix with infinity
    dtw_matrix = np.inf * np.ones((n, m))

    # Base case: dtw(0,0) = |signal1[0] - signal2[0]|
    dtw_matrix[0, 0] = abs(signal1[0] - signal2[0])

    # Fill the DTW matrix with the Sakoe-Chiba bound (window size = 1)
    for i in range(1, n):
        for j in range(max(0, i - window_size), min(m, i + window_size + 1)):
            cost = abs(signal1[i] - signal2[j])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],  # Insertion
                                          dtw_matrix[i, j - 1],  # Deletion
                                          dtw_matrix[i - 1, j - 1])  # Match

    # Backtrack to find the optimal path
    path = []
    i, j = n - 1, m - 1  # Start at the bottom-right corner
    while i > 0 or j > 0:
        path.append((i, j))
        # Move to the smallest neighbor (Insertion, Deletion, or Match)
        if i > 0 and j > 0:
            if dtw_matrix[i - 1, j] == min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]):
                i -= 1
            elif dtw_matrix[i, j - 1] == min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]):
                j -= 1
            else:
                i -= 1
                j -= 1
        elif i > 0:
            i -= 1
        else:
            j -= 1

    path.append((0, 0))  # Add the starting point
    path.reverse()  # Reverse the path to get the correct order

    return dtw_matrix, path, dtw_matrix[-1, -1]


# Define the signals
signal1 = np.array([3, 2, 0, 1, 4, 5, 6, 7, 2, 2, 1])
signal2 = np.array([5, 3, 1, 7, 8, 6, 9, 8, 6, 3, 2])

# Call the dtw function with Sakoe-Chiba bound (window size = 1)
dtw_matrix, path, distance = dtw_with_bound_margin1(signal1, signal2, window_size=1)


# Now, calculate the correct DTW distance (sum of the costs along the path)
# WHen Margin 1
dtw_distance = sum(abs(signal1[i] - signal2[j]) for i, j in path)


# Plot the DTW matrix with signals as x and y axes
plt.figure(figsize=(10, 6))
plt.imshow(dtw_matrix, cmap='Blues', origin='lower', interpolation='none', aspect='auto')

# Annotate the DTW matrix with the calculated values
for i in range(len(signal1)):
    for j in range(len(signal2)):
        plt.text(j, i, f'{dtw_matrix[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)

# Plot the optimal DTW path
plt.plot([p[1] for p in path], [p[0] for p in path], color='red', linewidth=2, marker='o', markersize=5)

# Set the axis labels based on the signals
plt.xticks(np.arange(len(signal2)), signal2, rotation=90)
plt.yticks(np.arange(len(signal1)), signal1)

# Add titles and labels
plt.title("DTW Matrix with Sakoe-Chiba Bound (Window Size 1) and Optimal Path")
plt.xlabel("Signal 2 (x-axis)")
plt.ylabel("Signal 1 (y-axis)")

# Show the plot
plt.show()

# Print the optimal path and its corresponding DTW values
# Compute the Euclidean distance (margin 0)
squared_diffs = (signal1 - signal2) ** 2
euclidean_distance = np.sqrt(np.sum(squared_diffs))
print(f"Euclidean Distance (Margin 0): {euclidean_distance:.5f}")

print("Optimal Path and Corresponding DTW Values:")

for (i, j) in path:
    print(f"Signal1[{i}] = {signal1[i]}, Signal2[{j}] = {signal2[j]} -> DTW Value: {dtw_matrix[i, j]:.2f}")
