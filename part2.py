import numpy as np
class NNClassifier:
    def __init__(self):
        self.train_data = None
        self.train_labels = None
    def train(self, features, labels):
        """Store training data."""
        self.train_data = features
        self.train_labels = labels
    def test(self, test_instance):
        """Predict the class label for a test instance."""
        distances = np.linalg.norm(self.train_data - test_instance, axis=1)
        nearest_index = np.argmin(distances)
        return self.train_labels[nearest_index]
class Validator:
    def __init__(self, classifier):
        self.classifier = classifier
    def leave_one_out_validation(self, features, labels):
        """Perform Leave-One-Out Validation."""
        correct_predictions = 0
        n = len(features)
        for i in range(n):
            # Leave out instance i as the test set
            test_instance = features[i]
            test_label = labels[i]
            # Use all other instances for training
            train_features = np.delete(features, i, axis=0)
            train_labels = np.delete(labels, i, axis=0)
            # Train and test
            self.classifier.train(train_features, train_labels)
            predicted_label = self.classifier.test(test_instance)
            if predicted_label == test_label:
                correct_predictions += 1
        # Calculate accuracy
        accuracy = correct_predictions / n
        return accuracy
def normalize_data(data):
    """Normalize features to range [0, 1]."""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)
def load_data(file_path):
    """Load dataset from file."""
    data = np.loadtxt(file_path)
    labels = data[:, 0]  # First column is class label
    features = data[:, 1:]  # Remaining columns are features
    return features, labels
if __name__ == "__main__":
    small_file_path = "small-test-dataset.txt"
    small_features, small_labels = load_data(small_file_path)
    small_features = normalize_data(small_features)
    small_selected_features = small_features[:, [2, 4, 6]]  # Features {3, 5, 7}
    small_classifier = NNClassifier()
    small_validator = Validator(small_classifier)
    small_accuracy = small_validator.leave_one_out_validation(small_selected_features, small_labels)
    print(f"Accuracy with features {3, 5, 7} on small-test-dataset.txt: {small_accuracy:.4f}")
    large_file_path = "large-test-dataset.txt"
    large_features, large_labels = load_data(large_file_path)
    large_features = normalize_data(large_features)
    large_selected_features = large_features[:, [0, 14, 26]]  # Features {1, 15, 27}
    large_classifier = NNClassifier()
    large_validator = Validator(large_classifier)
    large_accuracy = large_validator.leave_one_out_validation(large_selected_features, large_labels)
    print(f"Accuracy with features {1, 15, 27} on large-test-dataset.txt: {large_accuracy:.4f}")
