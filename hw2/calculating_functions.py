import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.linalg import svd
from sklearn.metrics import mean_squared_error


def linear_regression(X, y, gradient_function, alpha=0, l1_ratio=None, weights=None, learning_rate=0.01, epochs=1000,
                      show_progress=True):
    _X = np.c_[np.ones(X.shape[0]), X]

    if gradient_function == gradient_lasso:
        # z is the sum of squares of the elements of each column of X
        z = np.sum(_X * _X, axis=0)
    else:
        z = None

    # Initialize weights with zeros or provided initial weights
    if weights is None:
        weights = np.zeros(_X.shape[1])
    else:
        weights = np.array(weights)

    if show_progress:
        loop = tqdm(
            range(epochs),
            leave=True,
        )
    else:
        loop = range(epochs)

    for epoch in loop:
        # Calculate the gradient using the specified gradient function
        gradient = gradient_function(_X, y, weights, z, alpha, l1_ratio)
        if gradient_function == gradient_lasso:
            weights = gradient
        else:
            # Update weights
            weights -= learning_rate * gradient

    return weights


def gradient_ols(X, y, weights, z=None, alpha=0, l1_ratio=None):
    assert z is None, 'z is not used in OLS'
    assert alpha == 0, 'alpha is not used in OLS'
    assert l1_ratio is None, 'l1_ratio is not used in OLS'
    y_pred = np.dot(X, weights)
    gradient = -2 * np.dot(X.T, (y - y_pred))
    return gradient


def soft_threshold(rho, lamda, n):
    if rho < - lamda * n:
        return rho + (lamda * n)
    elif -lamda * n < rho < lamda * n:
        return 0
    else:
        return rho - (lamda * n)


# rho computation
def rho_compute_(y, X, w, j):
    # y is the response variable
    # X is the predictor variables matrix
    # w is the weight vector
    # j is the feature selector
    X_k = np.delete(X, j, 1)  # Remove the j variable i.e. j column
    w_k = np.delete(w, j)  # Remove the weight j
    predict_k = np.matmul(X_k, w_k)
    residual = y - predict_k
    rho_j = np.sum(X[:, j] * residual)
    return (rho_j)


def gradient_lasso(X_lasso, y, weights, z, alpha, l1_ratio=None):
    assert l1_ratio is None, 'l1_ratio is not used in Lasso'
    # for each independent variable
    for j in range(len(weights)):
        rho_j = rho_compute_(y, X_lasso, weights, j)
        # z is the sum of squares of the elements of the jth column of X
        if j == 0:
            weights[j] = rho_j / z[j]
        else:
            weights[j] = soft_threshold(rho_j, alpha, len(y)) / z[j]

    return weights


def gradient_ridge(X, y, weights, z=None, alpha=0.01, l1_ratio=None):
    assert z is None, 'z is not used in Ridge'
    assert l1_ratio is None, 'l1_ratio is not used in Ridge'
    y_pred = np.dot(X, weights)
    gradient = -2 * np.dot(X.T, (y - y_pred))
    gradient[1:] += 2 * alpha * weights[1:]
    return gradient


# def gradient_elastic_net(X, y, weights, z, alpha=0.01, l1_ratio=0.5):
#     y_pred = np.dot(X, weights)
#     # gradient = -2 * np.dot(X.T, (y - y_pred))
#     l2_term = 2 * alpha * (1 - l1_ratio) * weights[1:]
#     l1_term = l1_ratio * gradient_lasso(X, y, weights, z, alpha, None)
#     gradient = l1_ratio * l1_term
#     gradient[1:] += l2_term
#     return gradient

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))


def fit_calculate(X_test, y_test, weights, returnPred=False):
    isList = True
    # if weights is a simple numpy array, turn it into a list of one element
    if type(weights) == np.ndarray:
        weights = [weights]
        isList = False

    y_pred = [np.dot(X_test, w) for w in weights]
    mse = [mean_squared_error(y_test, y) for y in y_pred]
    r2 = [r2_score(y_test, y) for y in y_pred]

    if isList is False:
        mse = mse[0]
        r2 = r2[0]
        y_pred = y_pred[0]

    if returnPred:
        return mse, r2, y_pred
    else:
        return mse, r2


def incrementally_introduce_missing_values(original_df, additional_missing_percentage):
    """
    Introduce additional missing values into a copy of the dataset based on the additional missing percentage.

    :param original_df: Original DataFrame without missing values.
    :param additional_missing_percentage: Additional percentage of the original values to be made missing.
    :return: DataFrame with additional missing values introduced.
    """
    modified_df = original_df.copy()

    # Calculate the number of additional values to be made missing
    total_values = original_df.size
    num_additional_missing = int(np.floor(additional_missing_percentage * total_values))

    # Flatten the DataFrame and find indices of non-missing values
    flattened = modified_df.to_numpy().flatten()
    non_missing_indices = np.where(~np.isnan(flattened))[0]

    # Randomly select indices to introduce additional missing values
    additional_missing_indices = np.random.choice(non_missing_indices, num_additional_missing, replace=False)

    # Convert the 1D indices to 2D indices and introduce additional missing values
    row_indices, col_indices = np.unravel_index(additional_missing_indices, modified_df.shape)
    for row, col in zip(row_indices, col_indices):
        modified_df.iat[row, col] = np.nan

    return modified_df


def matrix_completion_nuclear_norm_2d(A, lambda_val=0.1, tolerance=1e-6, max_iterations=100):
    """
    Perform matrix completion using nuclear norm minimization, modified for 2D boolean array indexing.

    :param A: The input matrix with NaNs for missing values.
    :param lambda_val: Regularization parameter for nuclear norm.
    :param tolerance: Convergence tolerance.
    :param max_iterations: Maximum number of iterations.
    :return: Completed matrix.
    """
    # Initialize variables
    Y = np.nan_to_num(A)  # Replace NaNs with zeros
    mask = ~np.isnan(A)   # Mask of observed values
    norm_diff = np.inf
    iteration = 0

    # Iterative soft-thresholding
    while norm_diff > tolerance and iteration < max_iterations:
        U, S, Vt = svd(Y, full_matrices=False)
        S = np.maximum(S - lambda_val, 0)  # Soft thresholding
        Z = U @ np.diag(S) @ Vt            # Reconstruct matrix

        # Update Y only for missing values
        Y_prev = Y.copy()
        Y = np.where(mask, A, Z)  # Replace missing values with Z, keep original values from A

        # Check convergence
        norm_diff = np.linalg.norm(Y - Y_prev, 'fro') / np.linalg.norm(Y, 'fro')
        iteration += 1

    return Y


def impute(dataset, original_dataset, lambda_val=0.1, tolerance=1e-6, max_iterations=100):
    dataset_imputed = matrix_completion_nuclear_norm_2d(dataset, lambda_val, tolerance, max_iterations)
    mse = mean_squared_error(dataset_imputed.flatten(), original_dataset.values.flatten())
    return dataset_imputed, mse


# create a function that takes as input a dataframe with nan values
# computes the nuclear norm imputation for a range of lambdas given as input
# and returns the mse for each lambda
def nuclear_norm_imputation(df, original_df, lambdas):
    """
    Perform nuclear norm imputation for a range of lambda values.

    :param df: The input dataframe with NaNs for missing values.
    :param original_df: The original dataframe without missing values.
    :param lambdas: The range of lambda values to try.
    :return: List of MSE values for each lambda.
    """
    mse = []
    for l in tqdm(lambdas):
        df_imputed = matrix_completion_nuclear_norm_2d(df, lambda_val=l, tolerance=1e-6, max_iterations=100)
        df_imputed_flatten = df_imputed.flatten()
        original_flatten = original_df.values.flatten()
        mse.append(mean_squared_error(df_imputed_flatten, original_flatten))

    return mse
