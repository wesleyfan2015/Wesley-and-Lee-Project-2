import numpy as np
import time
# NNClassifier and Validator from Part II
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
def dataset_summary(features, labels):
    """Print dataset summary and default accuracy."""
    num_features = features.shape[1]
    num_instances = features.shape[0]
    print(f"This dataset has {num_features} features (not including the class attribute), "
          f"with {num_instances} instances.")
    print("\nPlease wait while I normalize the data... Done!")
    # Calculate default accuracy using leave-one-out validation with no features
    default_classifier = NNClassifier()
    validator = Validator(default_classifier)
    default_accuracy = validator.leave_one_out_validation(np.zeros((num_instances, 1)), labels) * 100
    print(f"\nRunning nearest neighbor with no features (default rate), using 'leave-one-out' evaluation, "
          f"I get an accuracy of {default_accuracy:.1f}%")
    print("\nBeginning search.")
    """"""
def forward_selection(features, labels):
    num_features = features.shape[1]
    current_set_of_features = []
    best_overall_accuracy = 0
    best_feature_subset = []
    for i in range(1, num_features + 1):
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0
        for k in range(1, num_features + 1):
            if k not in current_set_of_features:
                features_to_evaluate = current_set_of_features + [k]
                selected_features = features[:, [f - 1 for f in features_to_evaluate]]
                classifier = NNClassifier()
                validator = Validator(classifier)
                accuracy = validator.leave_one_out_validation(selected_features, labels) * 100
                print(f"   Using feature(s) {features_to_evaluate}, accuracy is {accuracy:.1f}%")
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        if feature_to_add_at_this_level is not None:
            current_set_of_features.append(feature_to_add_at_this_level)
            print(f"\nFeature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.1f}%")
            if best_so_far_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_so_far_accuracy
                best_feature_subset = list(current_set_of_features)
    print(f"\nFinished search!! The best feature subset is {best_feature_subset}, "
          f"which has an accuracy of {best_overall_accuracy:.1f}%")
def backward_elimination(features, labels):
    num_features = features.shape[1]
    current_set_of_features = list(range(1, num_features + 1))
    best_overall_accuracy = 0
    best_feature_subset = list(current_set_of_features)
    while len(current_set_of_features) > 1:
        feature_to_remove_at_this_level = None
        best_so_far_accuracy = 0
        for k in current_set_of_features:
            temp_features = current_set_of_features.copy()
            temp_features.remove(k)
            selected_features = features[:, [f - 1 for f in temp_features]]
            classifier = NNClassifier()
            validator = Validator(classifier)
            accuracy = validator.leave_one_out_validation(selected_features, labels) * 100
            print(f"   Using feature(s) {temp_features}, accuracy is {accuracy:.1f}%")
            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_remove_at_this_level = k
        if feature_to_remove_at_this_level is not None:
            current_set_of_features.remove(feature_to_remove_at_this_level)
            print(f"\nFeature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.1f}%")
            if best_so_far_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_so_far_accuracy
                best_feature_subset = list(current_set_of_features)
    print(f"\nFinished search!! The best feature subset is {best_feature_subset}, "
          f"which has an accuracy of {best_overall_accuracy:.1f}%")
def wesleys_special_algorithm(features, labels):
    num_features = features.shape[1]
    current_set_of_features = []
    best_overall_accuracy = 0
    best_feature_subset = []
    accuracy_cache = {}
    for level in range(1, num_features + 1):
        feature_to_add = None
        feature_to_remove = None
        best_so_far_accuracy = 0
        print(f"Level {level} of the search:")
        # Step 1: Forward - Try adding each feature not already in the set
        for k in range(1, num_features + 1):
            if k not in current_set_of_features:
                temp_features = current_set_of_features + [k]
                temp_features_tuple = tuple(sorted(temp_features))
                if temp_features_tuple not in accuracy_cache:
                    selected_features = features[:, [f - 1 for f in temp_features]]
                    classifier = NNClassifier()
                    validator = Validator(classifier)
                    accuracy = validator.leave_one_out_validation(selected_features, labels) * 100
                    accuracy_cache[temp_features_tuple] = accuracy
                else:
                    accuracy = accuracy_cache[temp_features_tuple]
                print(f"   Adding feature {k}, feature(s) {temp_features}, accuracy is {accuracy:.1f}%")
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = k
        # Step 2: Backward - Try removing each feature already in the set
        if len(current_set_of_features) > 2:  # Only attempt removal if more than 2 features
            for k in current_set_of_features:
                temp_features = [f for f in current_set_of_features if f != k]
                temp_features_tuple = tuple(sorted(temp_features))
                if temp_features_tuple not in accuracy_cache:
                    selected_features = features[:, [f - 1 for f in temp_features]]
                    classifier = NNClassifier()
                    validator = Validator(classifier)
                    accuracy = validator.leave_one_out_validation(selected_features, labels) * 100
                    accuracy_cache[temp_features_tuple] = accuracy
                else:
                    accuracy = accuracy_cache[temp_features_tuple]
                print(f"   Removing feature {k}, feature(s) {temp_features}, accuracy is {accuracy:.1f}%")
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = None  # If removal is better, skip addition
                    feature_to_remove = k
        # Step 3: Perform the best action
        if feature_to_add is not None:
            current_set_of_features.append(feature_to_add)
            print(f"\nFeature {feature_to_add} added. Feature set {current_set_of_features}, accuracy is {best_so_far_accuracy:.1f}%\n")
        elif feature_to_remove is not None:
            current_set_of_features.remove(feature_to_remove)
            print(f"\nFeature {feature_to_remove} removed. Feature set {current_set_of_features}, accuracy is {best_so_far_accuracy:.1f}%\n")
        else:
            print("No further improvement. Stopping search.")
            break
        # Update overall best if necessary
        if best_so_far_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_so_far_accuracy
            best_feature_subset = list(current_set_of_features)
    print(f"\nFinished search!! The best feature subset is {best_feature_subset}, which has an accuracy of {best_overall_accuracy:.1f}%")
    return best_feature_subset, best_overall_accuracy
def main():
    print("Welcome to Wesley and Lee's Feature Selection Algorithm.")
    # string = "titanic clean.txt"
    # string = "small-test-dataset.txt"
    string = "large-test-dataset.txt"
    # file_path = input("Enter the dataset file path: ")
    print("Enter the dataset file path: ",string)
    print("\nPlease wait while I normalize the data...")
    features, labels = load_data(string)
    features = normalize_data(features)
    print("Done!")
    num_features = features.shape[1]
    num_instances = features.shape[0]
    dataset_summary(features, labels)
    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print("3) Wesley and Lee's Special Algorithm")
    choice = int(input())
    if choice == 1:
        print("\nRunning Forward Selection...\n")
        start_time = time.time()
        forward_selection(features, labels)
        end_time = time.time()
        print(f"\nTime taken for Forward Selection: {end_time - start_time:.4f} seconds\n")
    elif choice == 2:
        print("\nRunning Backward Elimination...\n")
        start_time = time.time()
        backward_elimination(features, labels)
        end_time = time.time()
        print(f"\nTime taken for Backward Elimination: {end_time - start_time:.4f} seconds\n")
    elif choice == 3:
        print("\nRunning Wesley and Lee's Special Algorithm...\n")
        start_time = time.time()
        wesleys_special_algorithm(features, labels)
        end_time = time.time()
        print(f"\nTime taken for Wesley and Lee's Special Algorithm: {end_time - start_time:.4f} seconds\n")
    else:
        print("Invalid choice. Exiting.")
if __name__ == "__main__":
    main()
