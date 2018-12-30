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
  for i in range(num_train):
    score = np.dot(X[i],W)
    
    score -= max(score)
    score = np.exp(score)
    softmax_sum = np.sum(score)
    
    score /= softmax_sum
    
    loss -= np.log(score[y[i]])
    
    for j in range(num_classes):
        if j!=y[i]:
            dW[:,j]+=score[j]*X[i]
        else:
            dW[:,j]+=(score[j]-1)*X[i]
  
  loss /= num_train
  loss += reg*np.sum(W*W)
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
  num_class = W.shape[1]
  scores = X.dot(W)  # N*C
  # np.max后会变成一维，可设置keepdims=True变为二维(N,1)
  # 防止指数爆炸
  scores-=np.max(scores,axis=1,keepdims=True)
  # 取指数
  scores=np.exp(scores)
  # 计算softmax
  scores/=np.sum(scores,axis=1,keepdims=True)
  # ds表示L对S求导
  ds = np.copy(scores)
  # qiyi - 1
  ds[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, ds)
  loss = scores[np.arange(num_train), y]
  # 计算Li
  loss =-np.log(loss).sum()
  # 计算所有loss除以N
  loss /= num_train
  # ds矩阵没有除以N，所以在这里补上，最后除以N，具体看(5)式
  dW /= num_train
  # 计算最终的大L
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

