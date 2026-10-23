from pathlib import Path
from helpers import load_csv_data

# Get the folder where this current script is saved (project1)
script_dir = Path(__file__).resolve().parent

# Point to the directory containing the CSV files
data_dir = script_dir / "data" / "dataset"

print("Loading data... (this can take 10-30 seconds using NumPy)")

# Unpack all 5 values returned by load_csv_data
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
    str(data_dir), sub_sample=False
)

print("Data loaded successfully!")
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")


# Load test data (depending on your specific helpers.py, test data might use a separate function or flag)
# If your helpers.py has a separate function for test data:
# from helpers import load_test_data
# x_test, ids_test = load_test_data(str(data_dir))


import numpy as np


def standardize_and_impute(x):
    # Replace NaN with column means
    col_means = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_means, inds[1])

    # Standardize features (mean=0, std=1)
    col_stds = np.std(x, axis=0)
    col_stds[col_stds == 0] = 1.0  # Prevent division by zero
    x_scaled = (x - col_means) / col_stds
    return x_scaled


x_train_clean = standardize_and_impute(x_train)
x_test_clean = standardize_and_impute(x_test)


# Create a simple 80/20 train/validation split
np.random.seed(42)
num_samples = x_train_clean.shape[0]
indices = np.random.permutation(num_samples)
split_idx = int(num_samples * 0.8)

train_idx, val_idx = indices[:split_idx], indices[split_idx:]

x_tr, y_tr = x_train_clean[train_idx], y_train[train_idx]
x_val, y_val = x_train_clean[val_idx], y_train[val_idx]


# Example using your logistic regression function
# (Check your specific function signature from lab templates, e.g., gradient descent or regularized)
from implementations import logistic_regression  # or reg_logistic_regression

# Choose hyperparameters (lambda, max_iters, gamma/learning_rate)
lambda_ = 0.01
initial_w = np.zeros(x_tr.shape[1])
max_iters = 100
gamma = 0.01

# Train the model
w, loss = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)

# Predict on validation set
# Logistic regression usually outputs probabilities or scores. Threshold at 0.
scores = x_val @ w
y_pred = np.where(scores >= 0, 1, -1)

# Compute local accuracy
accuracy = np.mean(y_pred == y_val)
print(f"Local Validation Accuracy: {accuracy * 100:.2f}%")


from helpers import create_csv_submission

# 1. Train on ALL training data
w_final, _ = logistic_regression(
    y_train, x_train_clean, np.zeros(x_train_clean.shape[1]), max_iters=200, gamma=0.01
)

# 2. Predict on test set
test_scores = x_test_clean @ w_final
y_pred_test = np.where(test_scores >= 0, 1, -1)

# 3. Create submission file
output_path = "submission.csv"
create_csv_submission(test_ids, y_pred_test, output_path)
print(f"Submission saved to {output_path}!")
