from rba.train.rba_train import rba_train
from rba.test.rba_test import rba_test
from rba.density_estimation import lr_density_estimation

import scipy.io
import numpy as np
import os
import torch
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

folder_path = './data/gaussian1'

x_1 = scipy.io.loadmat(os.path.join(folder_path, 'x_1.mat'))['x_1']
x_2 = scipy.io.loadmat(os.path.join(folder_path, 'x_2.mat'))['x_2']
y_1 = np.transpose(scipy.io.loadmat(os.path.join(folder_path, 'y_1.mat'))['y_1'])
y_2 = np.transpose(scipy.io.loadmat(os.path.join(folder_path, 'y_2.mat'))['y_2'])

n_row, n_col = x_1.shape

x_1 = torch.FloatTensor(x_1)
x_2 = torch.FloatTensor(x_2)
x_1_b = torch.cat((torch.ones((n_row, 1)), torch.FloatTensor(x_1)), dim = 1)
x_2_b = torch.cat((torch.ones((n_row, 1)), torch.FloatTensor(x_2)), dim = 1)
y_1 = torch.FloatTensor(np.where(y_1 == 1, 1, 0))
y_2 = torch.FloatTensor(np.where(y_2 == 1, 1, 0))

# source mean 
mu_s = [6, 6] 

# source variance
var_s = [[3, -2], [-2, 3]] 

# target mean
mu_t = [7, 7] 

# target variance
var_t = [[3, 2], [2, 3]] 

mvn_s = multivariate_normal(mu_s, var_s)
mvn_t = multivariate_normal(mu_t, var_t)

d_ss, d_st, d_ts, d_tt, pdf = lr_density_estimation(x_1, x_2, [0.1, 1, 10])

# print(d_ss / d_st)

# # Because we have expert knowledge on mu and var for both source and target, 
# # we can get the predicted probabilities for each data point under source and target distributions 
d_s = mvn_s.pdf(x_1)
d_t = mvn_t.pdf(x_1)

r_st = torch.Tensor(d_ss / d_st).unsqueeze(1).detach() # torch.Tensor(d_s / d_t).unsqueeze(1).detach() # torch.ones((x_1.shape[0], 1))

rba_model = rba_train(x_1, y_1, r_st, max_itr = 10000, lr = 0.01) # torch.Tensor(d_s / d_t).unsqueeze(1)

loss, preds, acc = rba_test(rba_model, x_2, y_2, r_st)
print(f"Target Loss: {loss}. Accuracy: {acc}")


maxs = 15
mins = -5

X_dim1, X_dim2 = np.meshgrid(np.arange(mins, maxs + 0.1, 0.1), np.arange(mins, maxs + 0.1, 0.1))

dim = int((maxs - mins) / 0.1 + 1)

prediction = np.zeros((dim, dim))

for i in range(dim):
    for j in range(dim):
        x_t = torch.FloatTensor([X_dim1[i, j], X_dim2[i, j]])
        outputs = rba_model(x_t, pdf(x_t) / (1 - pdf(x_t)))
        prediction[i, j] = outputs[0]

plt.imshow(prediction, cmap='Spectral', interpolation='nearest', origin='lower', extent=[-5, 15, -5, 15])

pos = x_1[np.array(y_1 == 1).flatten()]
neg = x_1[np.array(y_1 == 0).flatten()]

plt.scatter(pos[:,0], pos[:,1], marker="x", color="black", s = 7)
plt.scatter(neg[:,0], neg[:,1], marker="o", color="white", s = 7)
plt.title("RBA - First Order Features")
plt.savefig('figure-1_expert.png')
