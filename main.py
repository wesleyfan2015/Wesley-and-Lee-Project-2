import random


def forward_selection(num_features):
    current_set_of_features = []
    best_overall_accuracy = 0
    best_feature_subset = []

    print("\nBeginning search.")
    for i in range(1, num_features + 1):
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        for k in range(1, num_features + 1):
            if k not in current_set_of_features:
                accuracy = random.uniform(20.0, 100.0)  # Dummy evaluation function with random accuracy
                print(f"\n   Using feature(s) {current_set_of_features + [k]} accuracy is {accuracy:.1f}%")

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k

        if feature_to_add_at_this_level is not None:
            current_set_of_features.append(feature_to_add_at_this_level)
            print(f"\nFeature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.1f}%")

            if best_so_far_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_so_far_accuracy
                best_feature_subset = list(current_set_of_features)

    print(
        f"\nFinished search!! The best feature subset is {best_feature_subset}, which has an accuracy of {best_overall_accuracy:.1f}%")


def backward_elimination(num_features):
    current_set_of_features = list(range(1, num_features + 1))
    best_overall_accuracy = 0
    best_feature_subset = []

    print("\nBeginning search.")
    while len(current_set_of_features) > 1:
        feature_to_remove_at_this_level = None
        best_so_far_accuracy = 0

        for k in current_set_of_features:
            temp_features = current_set_of_features.copy()
            temp_features.remove(k)
            accuracy = random.uniform(20.0, 100.0)  # Dummy evaluation function with random accuracy
            print(f"\n   Using feature(s) {temp_features} accuracy is {accuracy:.1f}%")

            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_remove_at_this_level = k

        if feature_to_remove_at_this_level is not None:
            current_set_of_features.remove(feature_to_remove_at_this_level)
            print(f"\nFeature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.1f}%")

            if best_so_far_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_so_far_accuracy
                best_feature_subset = list(current_set_of_features)

    print(
        f"\nFinished search!! The best feature subset is {best_feature_subset}, which has an accuracy of {best_overall_accuracy:.1f}%")
