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

def plot_dtw_table(dtw_table, signal1, signal2):
    # Plot the DTW table with labels and grid
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
    ax.set_title("DTW Cost Matrix")

    # Show grid
    ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)

    # Annotate the DTW table with values
    for i in range(len(signal1)):
        for j in range(len(signal2)):
            ax.text(j, i, f"{dtw_table[i + 1, j + 1]:.1f}",
                    ha='center', va='center', color='white')

    plt.show()



# Example usage
signal1 = [3, 2, 0, 1,4,5, 6,7,2,2,1]
signal2 = [4, 2, 1, 0, 5, 5,7,7,3,2,1]

dtw_table = compute_dtw(signal1, signal2)

print("DTW Table:")
print(dtw_table)

# Plot the DTW table
plot_dtw_table(dtw_table, signal1, signal2)