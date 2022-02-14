import scipy.io
import numpy as np
import os
from sklearn import preprocessing
import torch
from rba.density_estimation import lr_density_estimation

folder_path = './data/iris' # TODO Change this

iris_train = scipy.io.loadmat(os.path.join(folder_path, 'iris_train.mat'))['iris_train']
iris_test = scipy.io.loadmat(os.path.join(folder_path, 'iris_test.mat'))['iris_test']

X_s = iris_train[:,0:-1]
y_s = iris_train[:, -1]
X_t = iris_test[:,0:-1]
y_t = iris_test[:,-1]


lb = preprocessing.LabelBinarizer()
lb.fit(y_s)

Y_s = lb.transform(y_s)
Y_t = lb.transform(y_t)

X_s = torch.FloatTensor(X_s).detach()
X_t = torch.FloatTensor(X_t).detach()
Y_s = torch.FloatTensor(Y_s)
Y_t = torch.FloatTensor(Y_t)

n_row, n_col = X_s.shape


d_ss, d_st, d_ts, d_tt = lr_density_estimation(X_s, X_t, [0.1, 1, 10])