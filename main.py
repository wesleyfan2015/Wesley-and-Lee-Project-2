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

def wesleys_special_algorithm(num_features):
    current_set_of_features = []
    best_overall_accuracy = 0
    best_feature_subset = []

    print("\nBeginning Wesley's Special Algorithm search.")
    for i in range(1, num_features + 1):
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        for k in range(1, num_features + 1):
            if k not in current_set_of_features:
                accuracy = random.uniform(20.0, 100.0)
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

        if len(current_set_of_features) > 1:
            feature_to_remove_at_this_level = None
            best_after_removal_accuracy = best_so_far_accuracy

            for k in current_set_of_features:
                temp_features = current_set_of_features.copy()
                temp_features.remove(k)
                accuracy = random.uniform(20.0, 100.0)
                print(f"\n   After removing feature {k}, feature(s) {temp_features} accuracy is {accuracy:.1f}%")

                if accuracy > best_after_removal_accuracy:
                    best_after_removal_accuracy = accuracy
                    feature_to_remove_at_this_level = k

            if feature_to_remove_at_this_level is not None:
                current_set_of_features.remove(feature_to_remove_at_this_level)
                print(
                    f"\nFeature set {current_set_of_features} after removal was best, accuracy is {best_after_removal_accuracy:.1f}%")

                if best_after_removal_accuracy > best_overall_accuracy:
                    best_overall_accuracy = best_after_removal_accuracy
                    best_feature_subset = list(current_set_of_features)

    print(
        f"\nFinished search!! The best feature subset is {best_feature_subset}, which has an accuracy of {best_overall_accuracy:.1f}%")


def main():
    print("Welcome to Wesley and Lee's Feature Selection Algorithm.")
    num_features = int(input("Please enter total number of features: "))
    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print("3) Wesley and Lee's Special Algorithm")
    choice = int(input())

    if choice == 1:
        forward_selection(num_features)
    elif choice == 2:
        backward_elimination(num_features)
    elif choice == 3:
        wesleys_special_algorithm(num_features)
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
