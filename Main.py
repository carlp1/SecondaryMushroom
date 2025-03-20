import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
from Hyperparameter_Tuning import Node, DecisionTree



# Load the dataset
data = pd.read_csv("~/Downloads/secondary_data.csv", delimiter=';')

# Drop columns with >50% missing values
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

# Convert to NumPy arrays
X_np = X.values
y_np = y.values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# Initialize and train the DecisionTree with the specified configuration
tree = DecisionTree(
    criterion='gini',
    max_depth=15,
    min_samples_split=50,
    impurity_threshold=0.25
)
tree.fit(X_train, y_train)

# Predictions on training and test sets
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

# Calculate zero-one loss for both sets
train_loss = tree.zero_one_loss(y_train, y_train_pred)
test_loss = tree.zero_one_loss(y_test, y_test_pred)

# Record tree depth and number of leaves
tree_depth = tree.tree_depth
leaf_count = tree.get_leaf_count()

# Print metrics
print("\n=== Model Evaluation ===")
print(f"Training Loss: {train_loss:.5f}")
print(f"Testing Loss:  {test_loss:.5f}")
print(f"Tree Depth:    {tree_depth}")
print(f"Leaf Count:    {leaf_count}")

# Print the confusion matrix for the test set
print("\nConfusion Matrix for the Test Set:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Visualize the decision tree
print("\nVisualizing the Decision Tree...")
dot = tree.visualize_tree()
dot.format = 'png'
dot.render('decision_tree_visualization', view=True)