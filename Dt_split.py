import numpy as np

# Dataset
data = [
    {"HSV": 0.75, "WT": 24.0, "Species": "clownfish"},
    {"HSV": 0.62, "WT": 28.5, "Species": "tuna"},
    {"HSV": 0.83, "WT": 19.0, "Species": "angelfish"},
    {"HSV": 0.70, "WT": 26.0, "Species": "clownfish"},
    {"HSV": 0.58, "WT": 27.5, "Species": "tuna"},
    {"HSV": 0.85, "WT": 20.0, "Species": "angelfish"},
    {"HSV": 0.77, "WT": 25.0, "Species": "clownfish"},
    {"HSV": 0.61, "WT": 29.0, "Species": "tuna"},
]

# Helper functions
def calculate_entropy(subset):
    """Calculate entropy of a dataset subset."""
    total = len(subset)
    if total == 0:
        return 0
    species_counts = {}
    for item in subset:
        species_counts[item["Species"]] = species_counts.get(item["Species"], 0) + 1
    entropy = -sum((count / total) * np.log2(count / total) for count in species_counts.values())
    return entropy

def split_data(data, condition):
    """Split data based on a condition."""
    group1 = [item for item in data if condition(item)]
    group2 = [item for item in data if not condition(item)]
    return group1, group2

def calculate_information_gain(data, group1, group2):
    """Calculate Information Gain given original data and split groups."""
    total_entropy = calculate_entropy(data)
    weighted_entropy = (
        len(group1) / len(data) * calculate_entropy(group1) +
        len(group2) / len(data) * calculate_entropy(group2)
    )
    return total_entropy - weighted_entropy

# Split A: HSV > 0.70
group1_A, group2_A = split_data(data, lambda x: x["HSV"] > 0.70)
IG_A = calculate_information_gain(data, group1_A, group2_A)

# Split B: Water Temperature < 25.0
group1_B, group2_B = split_data(data, lambda x: x["WT"] < 25.0)
IG_B = calculate_information_gain(data, group1_B, group2_B)

# Output Results
print(f"Information Gain for Split A (HSV > 0.70): {IG_A:.8f}")
print(f"Information Gain for Split B (WT < 25.0): {IG_B:.8f}")

# Determine which split is better
if IG_A > IG_B:
    print("Split A is better.")
elif IG_B > IG_A:
    print("Split B is better.")
else:
    print("Both splits are equally useful.")
