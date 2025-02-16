import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Load the dataset
df = pd.read_csv("/Users/ariannagirotto/Desktop/SECONDO TRIMESTRE/MushroomDataset/secondary_data.csv", sep=";")

# Split into train and test sets (80% train, 20% test)
train_size = 0.8
np.random.seed(42)
indices = np.random.permutation(len(df))
train_indices = indices[:int(train_size * len(df))]
test_indices = indices[int(train_size * len(df)):]

df_train, df_test = df.iloc[train_indices], df.iloc[test_indices]
print(df_train)
print(df_test)

# Data Cleaning (on the training set first)
# Handle missing data and columns with too many missing values
missing_percent_train = df_train.isnull().mean()
columns_to_drop_train = missing_percent_train[missing_percent_train > 0.4].index
df_train_cleaned = df_train.drop(columns=columns_to_drop_train).dropna().drop_duplicates()
print(df_train_cleaned)

# Apply the same column removal to the test set (columns that exist in the cleaned training set)
df_test_cleaned = df_test[df_train_cleaned.columns].dropna().drop_duplicates()
print(df_test_cleaned)

train_size_after = len(df_train_cleaned) / (len(df_train_cleaned) + len(df_test_cleaned))
test_size_after = len(df_test_cleaned) / (len(df_train_cleaned) + len(df_test_cleaned))

print(f"Proportion after cleaning - Train: {train_size_after:.2%}, Test: {test_size_after:.2%}")

# Now, split the cleaned data into features (X) and target (y) for both train and test
X_train_cleaned = df_train_cleaned.drop("class", axis=1)
y_train_cleaned = (df_train_cleaned["class"] == "p").astype(int).values

X_test_cleaned = df_test_cleaned.drop("class", axis=1)
y_test_cleaned = (df_test_cleaned["class"] == "p").astype(int).values

# Identify categorical and numerical features after data cleaning
categorical_features = X_train_cleaned.select_dtypes(include=['object']).columns.tolist()
numerical_features = X_train_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Final cleaned features and target variables
X_train = X_train_cleaned
y_train = y_train_cleaned

X_test = X_test_cleaned
y_test = y_test_cleaned

# Check the distribution of 0s and 1s in the target variable for both train and test sets
train_class_distribution = np.bincount(y_train_cleaned)
test_class_distribution = np.bincount(y_test_cleaned)

# Print the distribution of classes (0s and 1s) in both train and test sets
print(f"Train class distribution (0s, 1s): {train_class_distribution}")
print(f"Test class distribution (0s, 1s): {test_class_distribution}")

# Check if the columns of the train set are equale to the columns to the test set to see if there are errors
train_columns = df_train_cleaned.columns
test_columns = df_test_cleaned.columns

# Find the column that are in the train and not in test set
missing_in_test = set(train_columns) - set(test_columns)

# Find the column that are in the test and not in train set
missing_in_train = set(test_columns) - set(train_columns)

# Shows the results
if missing_in_test:
    print(f"The following columns are in the train but not in test set: {missing_in_test}")
else:
    print("No missing column.")

if missing_in_train:
    print(f"The following columns are in the test but not in train set: {missing_in_train}")
else:
    print("No missing column.")


# IMPLEMENTATION OF THE ALGORITHMS


class TreeNode:
    def __init__(self, is_leaf=False, test=None, left=None, right=None, prediction=None):
        self.is_leaf = is_leaf
        self.test = test
        self.left = left
        self.right = right
        self.prediction = prediction


# Impurity functions
def gini_impurity(y):
    p = np.mean(y)
    return 2 * p * (1 - p)


def psi3(y):
    p = np.mean(y)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p / 2) * np.log2(p) - ((1 - p) / 2) * np.log2(1 - p)


def psi4(y):
    p = np.mean(y)
    return np.sqrt(p * (1 - p))


def select_impurity_function(impurity_type="gini"):
    if impurity_type == "gini":
        return gini_impurity
    elif impurity_type == "psi3":
        return psi3
    elif impurity_type == "psi4":
        return psi4
    else:
        raise ValueError("Unknown impurity function")


def split_dataset(X, y, feature, threshold):
    if feature in categorical_features:
        left_mask = X[feature] == threshold
    else:
        left_mask = X[feature] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]


def best_split(X, y, impurity_func):
    best_score = float('inf')
    best_test = None
    best_left_X, best_left_y, best_right_X, best_right_y = None, None, None, None

    for feature in X.columns:
        unique_values = X[feature].unique()
        thresholds = unique_values if feature in categorical_features else np.percentile(unique_values, [25, 50, 75])

        for threshold in thresholds:
            left_X, left_y, right_X, right_y = split_dataset(X, y, feature, threshold)

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            impurity = (len(left_y) / len(y)) * impurity_func(left_y) + (len(right_y) / len(y)) * impurity_func(right_y)

            if impurity < best_score:
                best_score = impurity
                best_test = (feature, threshold)
                best_left_X, best_left_y, best_right_X, best_right_y = left_X, left_y, right_X, right_y

    return best_test, best_left_X, best_left_y, best_right_X, best_right_y


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, impurity_type="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_func = select_impurity_function(impurity_type)
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return TreeNode(is_leaf=True, prediction=np.bincount(y).argmax())

        test, left_X, left_y, right_X, right_y = best_split(X, y, self.impurity_func)
        if test is None:
            return TreeNode(is_leaf=True, prediction=np.bincount(y).argmax())

        return TreeNode(is_leaf=False, test=test,
                        left=self._build_tree(left_X, left_y, depth + 1),
                        right=self._build_tree(right_X, right_y, depth + 1))

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for _, x in X.iterrows()])

    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.prediction
        feature, threshold = node.test
        if feature in categorical_features:
            branch = node.left if x[feature] == threshold else node.right
        else:
            branch = node.left if x[feature] <= threshold else node.right
        return self._predict_single(x, branch)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def error(self, X, y):
        return 1 - self.accuracy(X, y)


# Implement cross-validation
def cross_validation(X, y, param_grid, k=5):
    fold_size = len(X) // k
    results = []
    indices = np.random.permutation(len(X))

    for max_depth in param_grid["max_depth_list"]:
        for min_samples_split in param_grid["min_samples_split_list"]:
            for impurity in param_grid["impurity_list"]:
                fold_train_accuracies = []
                fold_test_accuracies = []
                fold_train_errors = []
                fold_test_errors = []

                for i in range(k):
                    test_indices = indices[i * fold_size:(i + 1) * fold_size]
                    train_indices = np.setdiff1d(indices, test_indices)

                    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
                    y_train, y_test = y[train_indices], y[test_indices]

                    tree = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split,
                                        impurity_type=impurity)
                    tree.fit(X_train, y_train)

                    train_accuracy = tree.accuracy(X_train, y_train)
                    test_accuracy = tree.accuracy(X_test, y_test)
                    train_error = tree.error(X_train, y_train)
                    test_error = tree.error(X_test, y_test)

                    fold_train_accuracies.append(train_accuracy)
                    fold_test_accuracies.append(test_accuracy)
                    fold_train_errors.append(train_error)
                    fold_test_errors.append(test_error)

                results.append({
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "impurity": impurity,
                    "mean_train_accuracy": np.mean(fold_train_accuracies),
                    "mean_test_accuracy": np.mean(fold_test_accuracies),
                    "mean_train_error": np.mean(fold_train_errors),
                    "mean_test_error": np.mean(fold_test_errors),
                })

    return pd.DataFrame(results)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, impurity_type="gini", max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_type = impurity_type
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrapping: sample with replacement
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X.iloc[indices], y[indices]

            # Feature sampling
            if self.max_features is None:
                max_features = int(np.sqrt(len(X.columns)))
            else:
                max_features = self.max_features

            selected_features = np.random.choice(X.columns, max_features, replace=False)
            X_sample = X_sample[selected_features]

            # Train a decision tree on the sampled data
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                impurity_type=self.impurity_type)
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, selected_features))

    def predict(self, X):
        tree_preds = np.array([tree.predict(X[selected_features]) for tree, selected_features in self.trees])
        return np.round(tree_preds.mean(axis=0)).astype(int)  # Majority voting

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def error(self, X, y):
        return 1 - self.accuracy(X, y)




def cross_validation_rf(X, y, param_grid, k=5, n_jobs=-1):
    fold_size = len(X) // k
    indices = np.random.permutation(len(X))

    def evaluate_params(n_trees, max_depth, min_samples_split, impurity, max_features):
        fold_train_accuracies, fold_test_accuracies = [], []
        fold_train_errors, fold_test_errors = [], []

        for i in range(k):
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)

            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            forest = RandomForest(n_trees=n_trees, max_depth=max_depth,
                                  min_samples_split=min_samples_split, impurity_type=impurity,
                                  max_features=max_features)
            forest.fit(X_train, y_train)

            train_accuracy = forest.accuracy(X_train, y_train)
            test_accuracy = forest.accuracy(X_test, y_test)

            fold_train_accuracies.append(train_accuracy)
            fold_test_accuracies.append(test_accuracy)
            fold_train_errors.append(1 - train_accuracy)
            fold_test_errors.append(1 - test_accuracy)

        return {
            "n_trees": n_trees,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "impurity": impurity,
            "max_features": max_features,
            "mean_train_accuracy": np.mean(fold_train_accuracies),
            "mean_test_accuracy": np.mean(fold_test_accuracies),
            "mean_train_error": np.mean(fold_train_errors),
            "mean_test_error": np.mean(fold_test_errors),
        }

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(n_trees, max_depth, min_samples_split, impurity, max_features)
        for n_trees in param_grid["n_trees_list"]
        for max_depth in param_grid["max_depth_list"]
        for min_samples_split in param_grid["min_samples_split_list"]
        for impurity in param_grid["impurity_list"]
        for max_features in param_grid["max_features_list"]
    )

    return pd.DataFrame(results)


# RESULTS


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


# DECISION TREE


# Initialization
tree = DecisionTree(max_depth=10, min_samples_split=2, impurity_type="gini")
tree.fit(X_train, y_train)

# Evaluation
test_accuracy_initial = tree.accuracy(X_test, y_test)
test_error_initial = tree.error(X_test, y_test)
train_accuracy_initial = tree.accuracy(X_train, y_train)
train_error_initial = tree.error(X_train, y_train)

print(f"Initial Train Accuracy: {train_accuracy_initial:.4f}")
print(f"Initial Train Error: {train_error_initial:.4f}")
print(f"Initial Test Accuracy: {test_accuracy_initial:.4f}")
print(f"Initial Test Error: {test_error_initial:.4f}")


# Hyperparameters
param_grid_dt = {
    "max_depth_list": [10, 15],
    "min_samples_split_list": [2, 10],
    "impurity_list": ["gini", "psi3", "psi4"]
}


# CROSS VALIDATION


results_dt_cv = cross_validation(X_train, y_train, param_grid_dt, k=5)
results_dt_cv_sorted = results_dt_cv.sort_values("mean_test_error", ascending=False)

print("\nCross-Validation Results (Sorted by Test Error):")
print(results_dt_cv_sorted[["max_depth", "min_samples_split", "impurity", "mean_train_accuracy", "mean_test_accuracy",
                            "mean_train_error", "mean_test_error"]])

# Best parameters for the Decision Tree ( the oens with the lowest test error)
best_params_dt = results_dt_cv.sort_values("mean_test_error").iloc[0]

print("\nBest Hyperparameters from Cross-Validation:")
print(f"Max Depth: {int(best_params_dt['max_depth'])}")
print(f"Min Samples Split: {int(best_params_dt['min_samples_split'])}")
print(f"Impurity Type: {best_params_dt['impurity']}")

# Final Best Model
final_tree_dt = DecisionTree(
    max_depth=int(best_params_dt["max_depth"]),
    min_samples_split=int(best_params_dt["min_samples_split"]),
    impurity_type=best_params_dt["impurity"]
)

final_tree_dt.fit(X_train, y_train)

# Evaluating the Best Model
final_test_accuracy_dt = final_tree_dt.accuracy(X_test, y_test)
final_test_error_dt = final_tree_dt.error(X_test, y_test)
final_train_accuracy_dt = final_tree_dt.accuracy(X_train, y_train)
final_train_error_dt = final_tree_dt.error(X_train, y_train)

print(f"\nFinal Model Results (after Hyperparameter Tuning):")
print(f"Final Test Accuracy: {final_test_accuracy_dt:.4f}")
print(f"Final Test Error: {final_test_error_dt:.4f}")
print(f"Final Train Accuracy: {final_train_accuracy_dt:.4f}")
print(f"Final Train Error: {final_train_error_dt:.4f}")


# RANDOM FOREST


rf = RandomForest(n_trees=5, max_depth=10, min_samples_split=2, impurity_type="gini", max_features=None)
rf.fit(X_train, y_train)

# Evaluation
test_accuracy_initial = rf.accuracy(X_test, y_test)
test_error_initial = rf.error(X_test, y_test)
train_accuracy_initial = rf.accuracy(X_train, y_train)
train_error_initial = rf.error(X_train, y_train)

print(f"Initial Train Accuracy: {train_accuracy_initial:.4f}")
print(f"Initial Train Error: {train_error_initial:.4f}")
print(f"Initial Test Accuracy: {test_accuracy_initial:.4f}")
print(f"Initial Test Error: {test_error_initial:.4f}")

# HYPERPARAMETERS
param_grid_rf = {
    "n_trees_list": [5],
    "max_depth_list": [10, 15],
    "min_samples_split_list": [2, 10],
    "impurity_list": ["gini", "psi3", "psi4"],
    "max_features_list": [3, 14]
}

# CROSS VALIDATION


results_rf_cv = cross_validation_rf(X_train, y_train, param_grid_rf, k=5, n_jobs=-1)
results_rf_cv_sorted = results_rf_cv.sort_values("mean_test_error", ascending=True)

print("\nCross-Validation Results (Sorted by Test Error):")
print(results_rf_cv_sorted[["n_trees", "max_depth", "min_samples_split", "impurity", "max_features",
                            "mean_train_accuracy", "mean_test_accuracy", "mean_train_error", "mean_test_error"]])

# Best Parameters for the Random Forest
best_params_rf = results_rf_cv_sorted.iloc[0]

print("\nBest Hyperparameters from Cross-Validation:")
print(f"Number of Trees: {int(best_params_rf['n_trees'])}")
print(f"Max Depth: {int(best_params_rf['max_depth'])}")
print(f"Min Samples Split: {int(best_params_rf['min_samples_split'])}")
print(f"Impurity Type: {best_params_rf['impurity']}")
print(f"Max Features: {best_params_rf['max_features']}")

# Best Final Model
final_rf = RandomForest(
    n_trees=int(best_params_rf["n_trees"]),
    max_depth=int(best_params_rf["max_depth"]),
    min_samples_split=int(best_params_rf["min_samples_split"]),
    impurity_type=best_params_rf["impurity"],
    max_features=best_params_rf["max_features"]
)

final_rf.fit(X_train, y_train)

# Evaluate the Best Model
final_test_accuracy_rf = final_rf.accuracy(X_test, y_test)
final_test_error_rf = final_rf.error(X_test, y_test)
final_train_accuracy_rf = final_rf.accuracy(X_train, y_train)
final_train_error_rf = final_rf.error(X_train, y_train)

print("\nFinal Model Results (after Hyperparameter Tuning):")
print(f"Final Test Accuracy: {final_test_accuracy_rf:.4f}")
print(f"Final Test Error: {final_test_error_rf:.4f}")
print(f"Final Train Accuracy: {final_train_accuracy_rf:.4f}")
print(f"Final Train Error: {final_train_error_rf:.4f}")
