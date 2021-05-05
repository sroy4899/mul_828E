import numpy as np
import os

pwd = os.path.dirname(os.path.realpath(__file__))

train_data = np.load(os.path.join(pwd, 'purchase2_train.npy'), allow_pickle=True)
test_data = np.load(os.path.join(pwd, 'purchase2_test.npy'), allow_pickle=True)

train_data = train_data.reshape((1,))[0]
test_data = test_data.reshape((1,))[0]

X_train = train_data['X'].astype(np.float32)
X_test = test_data['X'].astype(np.float32)
y_train = train_data['y'].astype(np.int64)
y_test = test_data['y'].astype(np.int64)

def load(indices, category='train'):
    if category == 'train':
        if max(indices) < len(X_train) and max(indices) < len(y_train):
            return X_train[indices], y_train[indices]
        else:
            l = np.array([a for a in indices if  a < len(X_train) and a < len(y_train)],np.int64)
            return X_train[l], y_train[l]
    elif category == 'test':
        return X_test[indices], y_test[indices]
