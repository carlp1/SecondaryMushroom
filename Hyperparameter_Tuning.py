import numpy as np
import math
import pandas as pd
from sklearn.model_selection import KFold
from graphviz import Digraph
from joblib import Parallel, delayed


# Node Class Definition

class Node:
    """
    A node in the decision tree.
    If is_leaf is True, it holds a class label.
    Otherwise, it defines a split (feature, threshold).
    """

    def __init__(self, feature=None, threshold=None, value=None, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.value = value  # If leaf, this holds the class label
        self.depth = depth  # Depth of the node in the tree
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.value is not None


# DecisionTree Class Definition

class DecisionTree:
    def __init__(self,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 impurity_threshold=None,
                 feature_names=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_threshold = impurity_threshold
        self.feature_names = feature_names
        self.root = None
        self.tree_depth = 0

        # Select impurity function
        if criterion == 'gini':
            self._impurity_func = self._gini
        elif criterion == 'entropy':
            self._impurity_func = self._entropy
        elif criterion == 'squared':
            self._impurity_func = self._squared
        else:
            raise ValueError("Unknown criterion: choose 'gini', 'entropy', or 'squared'.")
    #Builds the decision tree based on the training data.
    def fit(self, X, y):
        self.root = self.grow_tree(X, y, depth=0)
    #Recursively grows the tree by finding the best splits.
    def grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        current_impurity = self._impurity_func(y)

        self.tree_depth = max(self.tree_depth, depth)

        if len(np.unique(y)) == 1 or (num_samples < self.min_samples_split):
            return Node(value=self._majority_label(y), depth=depth)

        if (self.max_depth is not None) and (depth >= self.max_depth):
            return Node(value=self._majority_label(y), depth=depth)

        if (self.impurity_threshold is not None) and (current_impurity < self.impurity_threshold):
            return Node(value=self._majority_label(y), depth=depth)

        best_gain = 0
        best_feat = None
        best_thresh = None
        parent_impurity = current_impurity

        for feat_idx in range(num_features):
            col = X[:, feat_idx]
            for threshold in np.unique(col):
                gain = self._gain(col, y, threshold, parent_impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = threshold

        if best_gain == 0:
            return Node(value=self._majority_label(y), depth=depth)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        if not np.any(left_mask) or not np.any(right_mask):
            return Node(value=self._majority_label(y), depth=depth)

        left_node = self.grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self.grow_tree(X[right_mask], y[right_mask], depth + 1)

        node = Node(feature=best_feat, threshold=best_thresh, depth=depth)
        node.left = left_node
        node.right = right_node
        return node
    # Computes the impurity gain for a specific feature and threshold.

    def _gain(self, col, y, threshold, parent_imp):
        left_mask = (col <= threshold)
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            return 0

        y_left = y[left_mask]
        y_right = y[right_mask]
        n_left = len(y_left)
        n_right = len(y_right)
        n = len(y)

        left_imp = self._impurity_func(y_left)
        right_imp = self._impurity_func(y_right)
        child_imp = (n_left / n) * left_imp + (n_right / n) * right_imp
        return parent_imp - child_imp

    def _gini(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * math.log2(p) for p in probs if p > 0])

    def _squared(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return np.sum([math.sqrt(p * (1 - p)) for p in probs])

    def _majority_label(self, y):
        return np.bincount(y).argmax()

    # Predicts class labels for a set of input samples.
    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])
    # Recursively traverses the tree to predict the label for a single sample.
    def _traverse(self, row, node):
        if node.is_leaf():
            return node.value
        if row[node.feature] <= node.threshold:
            return self._traverse(row, node.left)
        else:
            return self._traverse(row, node.right)

    def zero_one_loss(self, y_true, y_pred):
        return np.mean(y_true != y_pred)

    def get_leaf_count(self):
        return self._count_leaves(self.root)

    def _count_leaves(self, node):
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def visualize_tree(self):
        dot = Digraph()

        def add_edges(dot, nd):
            if nd is None:
                return
            if nd.is_leaf():
                dot.node(str(id(nd)), f"Class {nd.value}", shape='ellipse',
                         style='filled', fillcolor='lightgreen')
            else:
                lbl = f"Feature {nd.feature}"
                if (self.feature_names is not None) and (0 <= nd.feature < len(self.feature_names)):
                    lbl = self.feature_names[nd.feature]
                dot.node(str(id(nd)), f"{lbl} <= {nd.threshold:.2f}", shape='box',
                         style='filled', fillcolor='lightblue')
                if nd.left:
                    add_edges(dot, nd.left)
                    dot.edge(str(id(nd)), str(id(nd.left)), label="<=")
                if nd.right:
                    add_edges(dot, nd.right)
                    dot.edge(str(id(nd)), str(id(nd.right)), label=">")

        add_edges(dot, self.root)
        return dot

    def grid_search(X, y, param_grid, feature_names=None, n_splits=5, n_jobs=-1, sample_frac=0.1):
        """
        Performs grid search with cross-validation for all hyperparameters simultaneously.
        Args:
            X: Features (numpy array or pandas DataFrame).
            y: Target labels (numpy array or pandas Series).
            param_grid: Dictionary of hyperparameters and their possible values.
            feature_names: List of feature names (optional).
            n_splits: Number of folds for cross-validation.
            n_jobs: Number of jobs for parallel processing.
            sample_frac: Fraction of the dataset to sample for faster execution.
        Returns:
            List of dictionaries containing the results for each hyperparameter combination.
        """
        # Convert X, y to NumPy arrays if not already
        X = np.array(X)
        y = np.array(y)

        # Sample a subset of the dataset
        if sample_frac < 1.0:
            sample_size = int(len(X) * sample_frac)
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X[indices]
            y = y[indices]

        def evaluate_combination(params):
            train_losses = []
            test_losses = []
            tree_depths = []
            leaf_counts = []

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_index, test_index in kf.split(X):
                X_train_cv, X_test_cv = X[train_index], X[test_index]
                y_train_cv, y_test_cv = y[train_index], y[test_index]

                tree = DecisionTree(
                    criterion=params['criterion'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    impurity_threshold=params['impurity_threshold'],
                    feature_names=feature_names
                )
                tree.fit(X_train_cv, y_train_cv)

                # Predict on the training fold
                y_train_pred = tree.predict(X_train_cv)
                train_loss = tree.zero_one_loss(y_train_cv, y_train_pred)
                train_losses.append(train_loss)

                # Predict on the test fold
                y_test_pred = tree.predict(X_test_cv)
                test_loss = tree.zero_one_loss(y_test_cv, y_test_pred)
                test_losses.append(test_loss)

                # Record tree properties
                tree_depths.append(tree.tree_depth)
                leaf_counts.append(tree.get_leaf_count())

            avg_train_loss = np.mean(train_losses)
            avg_test_loss = np.mean(test_losses)
            avg_depth = np.mean(tree_depths)
            avg_leaf = np.mean(leaf_counts)

            print(f"Trained with {params} -> "
                  f"Avg Train Loss: {avg_train_loss:.5f}, Avg Test Loss: {avg_test_loss:.5f}, "
                  f"Depth: {avg_depth:.2f}, Leaves: {avg_leaf:.2f}")

            return {
                **params,
                'avg_train_loss': avg_train_loss,
                'avg_test_loss': avg_test_loss,
                'avg_tree_depth': avg_depth,
                'avg_leaf_count': avg_leaf
            }

        # Generate all combinations of hyperparameters
        from itertools import product
        param_combinations = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]
        print(
            f"\nTotal number of parameter combinations: {len(param_combinations)} x {n_splits}-fold CV = {n_splits * len(param_combinations)} runs\n")

        # Run grid search in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_combination)(params) for params in param_combinations
        )

        return results


if __name__ == '__main__':
    # -----------------------
    # Data Loading & Preprocessing
    # -----------------------
    data = pd.read_csv("~/Downloads/secondary_data.csv", delimiter=';')

    # Drop columns with >50% missing values.
    missing_ratio = data.isnull().mean()
    print("Missing ratio per column:")
    print(missing_ratio)
    cols_to_drop = missing_ratio[missing_ratio > 0.5].index
    data = data.drop(columns=cols_to_drop)
    print(f"Dropped columns: {list(cols_to_drop)}")

    # Replace remaining missing values with the string "Missing"
    data = data.fillna("Missing")

    # One-Hot Encoding for Categorical Features
    data_encoded = pd.get_dummies(data)

    # Separate features & target
    X = data_encoded.drop(columns=['class_p', 'class_e'])
    y = data_encoded['class_p']

    # Define the parameter grid for grid search
    param_grid = {
        'criterion': ['gini', 'entropy', 'squared'],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10, 25, 50, 100],
        'impurity_threshold': [0.0, 0.01, 0.1, 0.2, 0.25, 0.5]
    }

    # Perform grid search
    print("\n===== Performing Grid Search =====")
    results = DecisionTree.grid_search(
        X, y,
        param_grid=param_grid,
        feature_names=list(X.columns),
        sample_frac=0.1  # Use 10% of the dataset
    )

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    print("\nGrid Search Results:")
    print(results_df)