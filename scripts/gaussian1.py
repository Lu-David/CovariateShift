from rba.experiment import BivariateExperiment

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


experiment = BivariateExperiment(mu_s, var_s, mu_t, var_t)
experiment.x_1 = x_1
experiment.y_1 = y_1
experiment.x_2 = x_2
experiment.y_2 = y_2
experiment.train_all()
experiment.plot_all()



