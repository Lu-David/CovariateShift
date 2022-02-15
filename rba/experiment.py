from rba.train.rba_train import rba_train
from rba.train.log_train import log_train
from rba.test.rba_test import rba_test
from rba.test.log_test import log_test
from rba.density_estimation import get_kernel_density_estimator, get_mvn_estimator, get_lr_density_estimator, ones

import scipy.io
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from rba.synthetic_data import BivariateGaussian

from rba.synthetic_data import BivariateGaussian
from rba.synthetic_data import BivariateGaussian

class BivariateExperiment():

    def __init__(self, mu_s, var_s, mu_t, var_t):
        self.mu_s = mu_s
        self.var_s = var_s
        self.mu_t = mu_t 
        self.var_t = var_t
        self.boundary_degree = 1

        gaussian = BivariateGaussian(mu_s, var_s, mu_t, var_t, self.boundary_degree)
        self.x_1, self.y_1, self.x_2, self.y_2 = gaussian.gen_data()

        self.kernel_dr = get_kernel_density_estimator(self.x_1, self.x_2)
        self.mvn_dr = get_mvn_estimator(mu_s, var_s, mu_t, var_t)
        # lr_dr = get_lr_density_estimator(x_1, x_2)
        self.ones_dr = ones

        self.desc = f""
        
        self.models = []
    
    def train_all(self, dr_estimator):
        rba_model = rba_train(self.x_1, self.y_1, dr_estimator, max_itr = 10000, lr = 0.01) 
        iw_model = log_train(self.x_1, self.y_1, dr_estimator, max_itr = 10000, lr = 0.01) 
        log_model = log_train(self.x_1, self.y_1, self.ones_dr, max_itr = 10000, lr = 0.01) 
        self.models = [rba_model, iw_model, log_model]

    def test_all(self):
        if len(self.models) == 0:
            return 
        for model in self.models:
            loss, preds, acc = log_test(model, self.x_2, self.y_2)

    def _plot_model(self, ax, model):

        log, preds, acc = log_test(model, self.x_2, self.y_2) # log_test and rba_test are the same
        mean = torch.mean(self.x_1, axis=0)
        std = torch.std(self.x_1, axis=0)
        
        maxs = mean + 3 * std
        mins = mean - 3 * std

        res = 0.01
        X_dim1, X_dim2 = np.meshgrid(np.arange(mins[0], maxs[0] + res, res), np.arange(mins[1], maxs[1] + res, res))
        dims = X_dim1.shape

        coors = np.dstack((X_dim1, X_dim2))
        coors = torch.FloatTensor(coors.reshape((dims[0] * dims[1], -1)))

        model.eval()
        predictions = model(coors)
        predictions = torch.reshape(predictions, (dims[0], dims[1]))

        ax[0].imshow(predictions.detach().numpy(), cmap='Spectral', interpolation='nearest', origin='lower', extent=[mins[0], maxs[0], mins[1], maxs[1]])

        pos = self.x_1[np.array(self.y_1 == 1).flatten()]
        neg = self.x_1[np.array(self.y_1 == 0).flatten()]
        ax[0].scatter(pos[:,0], pos[:,1], marker="x", color="black", s = 7)
        ax[0].scatter(neg[:,0], neg[:,1], marker="o", color="white", s = 7)
        ax[0].set_title(f"Source_{model.__name__}")

        ax[1].imshow(predictions.detach().numpy(), cmap='Spectral', interpolation='nearest', origin='lower', extent=[mins[0], maxs[0], mins[1], maxs[1]])

        pos = self.x_2[np.array(self.y_2 == 1).flatten()]
        neg = self.x_2[np.array(self.y_2 == 0).flatten()]
        ax[1].scatter(pos[:,0], pos[:,1], marker="x", color="black", s = 7)
        ax[1].scatter(neg[:,0], neg[:,1], marker="o", color="white", s = 7)
        ax[1].set_title(f"Target_Acc={round(acc.item(), 2)}")

    def plot_all(self):
        if len(self.models) == 0:
            return 

        fig, axes = plt.subplots(2, len(self.models))
        desc = f"Bound_deg={self.boundary_degree}; mu_s={self.mu_s}; mu_t={self.mu_t}"
        fig.suptitle(desc)

        for i, model in enumerate(self.models):
            self._plot_model(axes[:, i], model)

        fig.savefig(f'experiment_{desc}.png')            


