import numpy as np
from random import shuffle
from past.builtins import xrange


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
        f = X[i].dot(W)
        f = f - np.max(f)
        # loss += -np.log(np.exp(f[y[i]])/np.sum(np.exp(f)))
        loss += - f[y[i]] + np.log(np.sum(np.exp(f)))#只有正确类影响loss
        for j in xrange(num_classes):#正确类产生的影响loss先对f求导，再用f对w求导，loss对f的个分量求导，对f求导结果要区分是否正确类  全导数
            softmax_output = np.exp(f[j]) / sum(np.exp(f))
            if j == y[i]:
                dW[:, j] += (-1 + softmax_output) * X[i]
            else:
                dW[:, j] += softmax_output * X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    f = X.dot(W)
    f = f - np.max(f,axis=1).reshape(-1,1)
    loss = np.sum(- f[range(num_train), y] + np.log(np.sum(np.exp(f), axis=1)))  # 只有正确类影响loss
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    softmax_output = np.exp(f) / np.sum(np.exp(f), axis=1).reshape(-1,1)
    softmax_output[range(num_train),y] -= 1
    dW = X.T.dot(softmax_output)
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
