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

    def grid_search(X, y, criteria, stop_criterion, stop_values, feature_names=None, n_splits=5, n_jobs=-1):
        """
        Performs grid search with cross validation for a single stopping criterion.
        For each impurity measure in 'criteria' and for each value in 'stop_values' for the given stopping criterion,
        this function does the following:
            - Splits the data into 'n_splits' folds,
            - For each fold, trains on the (n_splits-1) folds and tests on the remaining fold,
            - Computes both training and test zero-one loss,
            - Records the average performance across folds along with the average tree depth and number of leaves.
            - Uses parallelization via joblib to speed up the process across parameter combinations.
        """
        # Convert X, y to NumPy arrays if not already
        X = np.array(X)
        y = np.array(y)

        def evaluate_combination(crit, val):
            train_losses = []
            test_losses = []
            tree_depths = []
            leaf_counts = []

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_index, test_index in kf.split(X):
                X_train_cv, X_test_cv = X[train_index], X[test_index]
                y_train_cv, y_test_cv = y[train_index], y[test_index]

                if stop_criterion == 'max_depth':
                    tree = DecisionTree(
                        criterion=crit,
                        max_depth=val,
                        min_samples_split=2,
                        impurity_threshold=None,
                        feature_names=feature_names
                    )
                elif stop_criterion == 'min_samples_split':
                    tree = DecisionTree(
                        criterion=crit,
                        max_depth=None,
                        min_samples_split=val,
                        impurity_threshold=None,
                        feature_names=feature_names
                    )
                elif stop_criterion == 'impurity_threshold':
                    tree = DecisionTree(
                        criterion=crit,
                        max_depth=None,
                        min_samples_split=2,
                        impurity_threshold=val,
                        feature_names=feature_names
                    )
                else:
                    raise ValueError(
                        "stop_criterion must be one of: 'max_depth', 'min_samples_split', 'impurity_threshold'")

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

            print(f"Trained with {crit}, {stop_criterion}={val} -> "
                  f"Avg Train Loss: {avg_train_loss:.5f}, Avg Test Loss: {avg_test_loss:.5f}, "
                  f"Depth: {avg_depth:.2f}, Leaves: {avg_leaf:.2f}")

            return {
                'criterion': crit,
                stop_criterion: val,
                'avg_train_loss': avg_train_loss,
                'avg_test_loss': avg_test_loss,
                'avg_tree_depth': avg_depth,
                'avg_leaf_count': avg_leaf
            }

        param_combinations = [(crit, val) for crit in criteria for val in stop_values]
        print(
            f"\nTotal number of parameter combinations: {len(param_combinations)} x {n_splits}-fold CV = {n_splits * len(param_combinations)} runs\n")

        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_combination)(crit, val) for crit, val in param_combinations
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



    # Grid Search: Tuning max_depth
    print("\n===== Tuning max_depth =====")
    max_depth_values = [10, 11, 12, 13, 20, 25, 30, 40, 50]
    results_max_depth = DecisionTree.grid_search(
        X, y,
        criteria=['gini', 'entropy', 'squared'],
        stop_criterion='max_depth',
        stop_values=max_depth_values,
        feature_names=list(X.columns)
    )
    results_max_depth_df = pd.DataFrame(results_max_depth)
    # Sort by impurity measure first, then by max_depth
    results_max_depth_df = results_max_depth_df.sort_values(by=['criterion', 'max_depth'])
    print("\nResults for max_depth tuning:")
    print(results_max_depth_df)

    # Grid Search: Tuning min_samples_split
    print("\n===== Tuning min_samples_split =====")
    min_samples_values = [2, 5, 10, 25, 50, 100]
    results_min_samples = DecisionTree.grid_search(
        X, y,
        criteria=['gini', 'entropy', 'squared'],
        stop_criterion='min_samples_split',
        stop_values=min_samples_values,
        feature_names=list(X.columns)
    )
    results_min_samples_df = pd.DataFrame(results_min_samples)
    results_min_samples_df = results_min_samples_df.sort_values(by=['criterion', 'min_samples_split'])
    print("\nResults for min_samples_split tuning:")
    print(results_min_samples_df)

    # Grid Search: Tuning impurity_threshold
    print("\n===== Tuning impurity_threshold =====")
    impurity_threshold_values = [0.0, 0.01, 0.1, 0.2, 0.25, 0.5]
    results_impurity_thresh = DecisionTree.grid_search(
        X, y,
        criteria=['gini', 'entropy', 'squared'],
        stop_criterion='impurity_threshold',
        stop_values=impurity_threshold_values,
        feature_names=list(X.columns)
    )
    results_impurity_thresh_df = pd.DataFrame(results_impurity_thresh)
    results_impurity_thresh_df = results_impurity_thresh_df.sort_values(by=['criterion', 'impurity_threshold'])
    print("\nResults for impurity_threshold tuning:")
    print(results_impurity_thresh_df)
