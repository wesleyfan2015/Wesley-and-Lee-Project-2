import numpy as np
import matplotlib.pyplot as plt

# Function to load dataset
def load_dataset(file_path):
    data = np.loadtxt(file_path)
    labels = data[:, 0]  # First column is the class label
    features = data[:, 1:]  # Remaining columns are the features
    return features, labels

# Load datasets
small_features, small_labels = load_dataset("small-test-dataset.txt")
large_features, large_labels = load_dataset("large-test-dataset.txt")

# Choose feature indices based on separation ability
# Small Dataset: Feature 3 (index 2) vs. Feature 5 (index 4)
small_feature_x = small_features[:, 2]
small_feature_y = small_features[:, 4]

# Large Dataset: Feature 1 (index 0) vs. Feature 15 (index 14)
large_feature_x = large_features[:, 0]
large_feature_y = large_features[:, 14]

# Scatter plot for well-separating features
plt.figure(figsize=(12, 6))

# Small Dataset: Feature 3 vs Feature 5
plt.subplot(1, 2, 1)
plt.scatter(small_features[:, 2], small_features[:, 4], c=small_labels, cmap='coolwarm', alpha=0.7, edgecolor='k')
plt.title("Small Dataset: Feature 3 vs. Feature 5")
plt.xlabel("Feature 3")
plt.ylabel("Feature 5")
plt.colorbar(label="Class Labels")

# Large Dataset: Feature 1 vs Feature 15
plt.subplot(1, 2, 2)
plt.scatter(large_features[:, 0], large_features[:, 14], c=large_labels, cmap='coolwarm', alpha=0.7, edgecolor='k')
plt.title("Large Dataset: Feature 1 vs. Feature 15")
plt.xlabel("Feature 1")
plt.ylabel("Feature 15")
plt.colorbar(label="Class Labels")

plt.tight_layout()
plt.savefig("features_that_separate_classes_well.png")
plt.show()


# Scatter plot for poorly-separating features
plt.figure(figsize=(12, 6))

# Small Dataset: Feature 7 vs Feature 8
plt.subplot(1, 2, 1)
plt.scatter(small_features[:, 6], small_features[:, 7], c=small_labels, cmap='coolwarm', alpha=0.7, edgecolor='k')
plt.title("Small Dataset: Feature 7 vs. Feature 8")
plt.xlabel("Feature 7")
plt.ylabel("Feature 8")
plt.colorbar(label="Class Labels")

# Large Dataset: Feature 11 vs Feature 12
plt.subplot(1, 2, 2)
plt.scatter(large_features[:, 10], large_features[:, 11], c=large_labels, cmap='coolwarm', alpha=0.7, edgecolor='k')
plt.title("Large Dataset: Feature 11 vs. Feature 12")
plt.xlabel("Feature 11")
plt.ylabel("Feature 12")
plt.colorbar(label="Class Labels")

plt.tight_layout()
plt.savefig("features_that_dont_separate_classes_well.png")
plt.show()
