import math
import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in xrange(num_train):
    sum = 0
    yi = np.dot(X[i], W[:, y[i]])
    dW_i_j = np.zeros(W.shape)
    for j in xrange(num_classes):
        e_power_j = math.exp(np.dot(X[i], W[:, j]))
        sum += e_power_j
        dW_i_j[:, j] = X[i] * e_power_j
    loss += math.log(sum) - yi
    dW_i_j /= sum
    dW += dW_i_j
    dW[:, y[i]] -= X[i]
    
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW = dW / num_train + W * reg
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  n, d = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  fj = np.dot(X, W)
  fy = fj[np.arange(n), y]
  fj_exp = np.exp(fj)
  loss_sum = np.sum(fj_exp, axis = 1) # [n]
  loss = np.sum(np.log(loss_sum)) - np.sum(fy) 
  loss /= n
  loss += 0.5 * reg * np.sum(W * W)
    
  prob_matrix = fj_exp / loss_sum[:, None]
  prob_matrix[np.arange(n), y] -= 1
  dW = np.dot(X.T, prob_matrix) / n + W * reg
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

