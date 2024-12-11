import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


# Updated dataset as lists
A = [1, 1, 2, 2, 1, 0, 0, 1, 0, 2]
B = [2, 2, 3, 0, 3, 0, 1, 3, 2, 2]
C = [1, 3, 2, 3, 2, 3, 0, 1, 1, 3]
labels = [1, 0, 0, 1, 0, 1, 0, 0, 1, 1]


# Function to calculate entropy of a label distribution
def entropy(label_column):
    total = len(label_column)
    counts = Counter(label_column)
    return -sum((count / total) * np.log2(count / total) for count in counts.values() if count > 0)


# Function to calculate information gain for a feature
def information_gain(A, B, C, labels, feature):
    # Calculate the entropy of the entire dataset (root entropy)
    root_entropy = entropy(labels)

    # Split the dataset based on unique values of the feature
    if feature == 'A':
        unique_values = set(A)
        feature_values = A
    elif feature == 'B':
        unique_values = set(B)
        feature_values = B
    elif feature == 'C':
        unique_values = set(C)
        feature_values = C

    weighted_entropy = 0
    for value in unique_values:
        subset_labels = [labels[i] for i in range(len(labels)) if feature_values[i] == value]
        weighted_entropy += (len(subset_labels) / len(labels)) * entropy(subset_labels)

    # Information Gain is the reduction in entropy
    return root_entropy - weighted_entropy


# Compute Information Gain for Feature A and Feature B
ig_A = information_gain(A, B, C, labels, 'A')
ig_B = information_gain(A, B, C, labels, 'B')

print(f"Information Gain for Feature A: {ig_A:.6f}")
print(f"Information Gain for Feature B: {ig_B:.6f}")


# Function to calculate KNN and predict the label
def knn_predict(A, B, C, labels, input_vector, k=3):
    # Create a list of data points (features and labels)
    data_points = list(zip(A, B, C, labels))

    # Compute Euclidean distance between the input_vector and all data points
    distances = []
    for point in data_points:
        distance = np.sqrt(
            (point[0] - input_vector[0]) ** 2 + (point[1] - input_vector[1]) ** 2 + (point[2] - input_vector[2]) ** 2)
        distances.append((distance, point[3]))  # Store (distance, label)

    # Sort by distance and take the k nearest neighbors
    distances.sort(key=lambda x: x[0])
    nearest_neighbors = [label for _, label in distances[:k]]

    # Predict the label as the majority label of the k nearest neighbors
    predicted_label = Counter(nearest_neighbors).most_common(1)[0][0]

    return predicted_label


# Function to calculate Euclidean distance
def euclidean_distance(input_vector, point):
    return np.sqrt((input_vector[0] - point[0]) ** 2 +
                   (input_vector[1] - point[1]) ** 2 +
                   (input_vector[2] - point[2]) ** 2)


# Function to calculate Manhattan distance
def manhattan_distance(input_vector, point):
    return abs(input_vector[0] - point[0]) + abs(input_vector[1] - point[1]) + abs(input_vector[2] - point[2])


# Function to calculate KNN prediction
def knn_predict(A, B, C, labels, input_vector, k=3, distance_metric='euclidean'):
    # Create a list of data points (features and labels)
    data_points = list(zip(A, B, C, labels))

    # Compute distances based on the selected metric
    distances = []
    for point in data_points:
        if distance_metric == 'euclidean':
            distance = euclidean_distance(input_vector, point[:3])
        elif distance_metric == 'manhattan':
            distance = manhattan_distance(input_vector, point[:3])
        distances.append((distance, point[3]))  # Store (distance, label)

    # Sort by distance and take the k nearest neighbors
    distances.sort(key=lambda x: x[0])
    nearest_neighbors = [label for _, label in distances[:k]]

    # Predict the label as the majority label of the k nearest neighbors
    predicted_label = Counter(nearest_neighbors).most_common(1)[0][0]

    return predicted_label


# Function to calculate confusion matrix, precision, recall, and accuracy
def evaluate_classifier(y_true, y_pred):
    # Confusion matrix: TP, FP, FN, TN
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]  # True Positive
    FP = cm[0, 1]  # False Positive
    FN = cm[1, 0]  # False Negative
    TN = cm[0, 0]  # True Negative

    # Accuracy: (TP + TN) / total
    accuracy = accuracy_score(y_true, y_pred)

    # Precision for class 1 (cyberattack)
    precision_1 = precision_score(y_true, y_pred, pos_label=1)

    # Recall for class 1 (cyberattack)
    recall_1 = recall_score(y_true, y_pred, pos_label=1)

    # Precision for class 0 (non-cyberattack)
    precision_0 = precision_score(y_true, y_pred, pos_label=0)

    # Recall for class 0 (non-cyberattack)
    recall_0 = recall_score(y_true, y_pred, pos_label=0)

    return cm, accuracy, precision_0, recall_0, precision_1, recall_1


# Example dataset with actual labels (0 = non-attack, 1 = attack)
# Here we simulate the predictions as well.
y_true = [1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]

# Evaluate the classifier's performance
cm, accuracy, precision_0, recall_0, precision_1, recall_1 = evaluate_classifier(y_true, y_pred)

# Output confusion matrix and metrics
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy}")
print(f"Precision (class 0): {precision_0}")
print(f"Recall (class 0): {recall_0}")
print(f"Precision (class 1): {precision_1}")
print(f"Recall (class 1): {recall_1}")

# Input feature vector (1, 1, 1)
input_vector = [1, 1, 1]

# (b) KNN with Euclidean distance and K=3
predicted_label_euclidean = knn_predict(A, B, C, labels, input_vector, k=3, distance_metric='euclidean')
print(f"Predicted label for input vector (1, 1, 1) using Euclidean distance (K=3): {predicted_label_euclidean}")

# (c) KNN with Manhattan distance and K=5
predicted_label_manhattan = knn_predict(A, B, C, labels, input_vector, k=5, distance_metric='manhattan')
print(f"Predicted label for input vector (1, 1, 1) using Manhattan distance (K=5): {predicted_label_manhattan}")


