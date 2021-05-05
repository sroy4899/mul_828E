import numpy as np
import os
import pickle
from urllib import request
import gzip
pwd = os.path.dirname(os.path.realpath(__file__))

# lets just hack some stuff together. taken from mnist.py
# https://github.com/hsjeong5/MNIST-for-Numpy
def mnist():
    with open(os.path.join(pwd,"mnist.pkl"),'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

X_train, X_test, y_train, y_test = mnist()
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

def load(indices, category='train'):
    if category == 'train':
        if max(indices) < len(X_train) and max(indices) < len(y_train):
            return X_train[indices], y_train[indices]
        else:
            l = np.array([a for a in indices if  a < len(X_train) and a < len(y_train)],np.int64)
            return X_train[l], y_train[l]
    elif category == 'test':
        return X_test[indices], y_test[indices]

