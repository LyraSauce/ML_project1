import numpy as np
import os


def mean_squared_error(y, tx, w):
    """
    Compute the loss by MSE.

    Args:
        y (np.array): shape=(N,1)
        tx (np.array): shape=(N,D)
        w (np.array): shape=(D,1)

    Returns:
        float: the loss according to the MSE
    """
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


def sigmoid(t):
    """
    Compute the sigmoid function.

    Args:
        t (np.array): shape=(N,1)

    Returns:
        np.array: shape=(N,1)
    """
    return 1 / (1 + np.exp(-t))


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent

    Args:
        y (np.array): shape=(N,1)
        tx (np.array): shape=(N,D)
        initial_w (np.array): shape=(D,1)
        max_iters (int): the number of iterations
        gamma (float): the learning rate

    Returns:
        w (np.array): shape=(D,1)
        loss (float): the loss according to the MSE
    """
    w = initial_w
    for _ in range(max_iters):
        grad = -tx.T.dot(y - tx.dot(w)) / len(y)
        w = w - gamma * grad
    loss = mean_squared_error(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent

    Args:
        y (np.array): shape=(N,1)
        tx (np.array): shape=(N,D)
        initial_w (np.array): shape=(D,1)
        max_iters (int): the number of iterations
        gamma (float): the learning rate

    Returns:
        w (np.array): shape=(D,1)
        loss (float): the loss according to the MSE
    """
    w = initial_w
    for _ in range(max_iters):
        rand_idx = np.random.randint(0, len(y))
        x_i = tx[rand_idx : rand_idx + 1]
        y_i = y[rand_idx : rand_idx + 1]
        grad = -x_i.T.dot(y_i - x_i.dot(w))
        w = w - gamma * grad
    loss = mean_squared_error(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """
    Least squares regression

    Args:
        y (np.array): shape=(N,1)
        tx (np.array): shape=(N,D)

    Returns:
        w (np.array): shape=(D,1)
        loss (float): the loss according to the MSE
    """
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    loss = mean_squared_error(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations

    Args:
        y (np.array): shape=(N,1)
        tx (np.array): shape=(N,D)
        lambda_ (float): the regularization parameter

    Returns:
        w (np.array): shape=(D,1)
        loss (float): the loss according to the MSE
    """
    N = len(y)
    a = tx.T.dot(tx) + (2 * N * lambda_) * np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = mean_squared_error(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent


    Args:
        y (np.array): shape=(N,1)
        tx (np.array): shape=(N,D)
        initial_w (np.array): shape=(D,1)
        max_iters (int): the number of iterations
        gamma (float): the learning rate

    Returns:
        w (np.array): shape=(D,1)
        loss (float): the loss according to the log-likelihood
    """
    w = initial_w
    for _ in range(max_iters):
        pred = sigmoid(tx.dot(w))
        grad = tx.T.dot(pred - y) / len(y)
        w = w - gamma * grad
    pred = sigmoid(tx.dot(w))
    loss = -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) / len(y)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent

    Args:
        y (np.array): shape=(N,1)
        tx (np.array): shape=(N,D)
        lambda_ (float): the regularization parameter
        initial_w (np.array): shape=(D,1)
        max_iters (int): the number of iterations
        gamma (float): the learning rate

    Returns:
        w (np.array): shape=(D,1)
        loss (float): the loss according to the log-likelihood with regularization
    """
    w = initial_w
    for _ in range(max_iters):
        pred = sigmoid(tx.dot(w))
        grad = tx.T.dot(pred - y) / len(y) + lambda_ * w * 2
        w = w - gamma * grad
    pred = sigmoid(tx.dot(w))
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return w, loss
