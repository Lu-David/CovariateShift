from rba.train.rba_train import rba_train
from rba.test.rba_test import rba_test
from rba.density_estimation import get_kernel_density_estimator, get_mvn_estimator, get_lr_density_estimator
from rba.plot import two_dim_plot

import scipy.io
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


dirname = os.path.dirname(__file__)
folder_path = os.path.join(dirname,'../data/gaussian1')

x_1 = scipy.io.loadmat(os.path.join(folder_path, 'x_1.mat'))['x_1']
x_2 = scipy.io.loadmat(os.path.join(folder_path, 'x_2.mat'))['x_2']
y_1 = np.transpose(scipy.io.loadmat(os.path.join(folder_path, 'y_1.mat'))['y_1'])
y_2 = np.transpose(scipy.io.loadmat(os.path.join(folder_path, 'y_2.mat'))['y_2'])

n_row, n_col = x_1.shape

x_1 = torch.FloatTensor(x_1)
x_2 = torch.FloatTensor(x_2)
# x_1_b = torch.cat((torch.ones((n_row, 1)), torch.FloatTensor(x_1)), dim = 1)
# x_2_b = torch.cat((torch.ones((n_row, 1)), torch.FloatTensor(x_2)), dim = 1)
y_1 = torch.FloatTensor(np.where(y_1 == 1, 1, 0))
y_2 = torch.FloatTensor(np.where(y_2 == 1, 1, 0))

mu_s = [6, 6] 
var_s = [[3, -2], [-2, 3]] 
mu_t = [7, 7] 
var_t = [[3, 2], [2, 3]] 


"""
Get Density Ratio Estimators
"""

kernel_dr = get_kernel_density_estimator(x_1, x_2)
mvn_dr = get_mvn_estimator(mu_s, var_s, mu_t, var_t)
# lr_dr = get_lr_density_estimator(x_1, x_2)

"""
RBA Kernel
"""
r_st = torch.Tensor(kernel_dr(x_1)).unsqueeze(1)
rba_model = rba_train(x_1, y_1, r_st, max_itr = 10000, lr = 0.01) # torch.Tensor(d_s / d_t).unsqueeze(1)

loss, preds, acc = rba_test(rba_model, x_2, y_2, r_st)
print(f"Target Loss: {loss}. Target Accuracy: {acc}")
two_dim_plot(rba_model, kernel_dr, x_1, y_1)

# """
# RBA LR
# """
# r_st = torch.Tensor(lr_dr(x_1)).detach()
# rba_model = rba_train(x_1, y_1, r_st, max_itr = 10000, lr = 0.01) # torch.Tensor(d_s / d_t).unsqueeze(1)
# for param in rba_model.parameters():
#     print(param.data)
# loss, preds, acc = rba_test(rba_model, x_2, y_2, r_st)
# print(f"Target Loss: {loss}. Target Accuracy: {acc}")
# two_dim_plot("RBA_lr", rba_model, lr_dr, x_1, y_1)

