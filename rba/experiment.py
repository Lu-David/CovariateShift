from rba.train.rba_train import rba_train
from rba.train.log_train import log_train
from rba.test.rba_test import rba_test
from rba.test.log_test import log_test
from rba.density_estimation import get_kernel_density_estimator, get_mvn_estimator, get_lr_density_estimator, ones
from rba.plot import heatmap_model, scatter_binary, confidence_ellipse
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
        self.lr_dr = get_lr_density_estimator(self.x_1, self.x_2)
        self.ones_dr = ones
        
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

        x_1_np = self.x_1.detach().numpy()
        x_2_np = self.x_2.detach().numpy()

        heatmap_model(self.x_1, self.y_1, ax[0], model)
        scatter_binary(self.x_1, self.y_1, ax[0])
        confidence_ellipse(x_1_np[:, 0], x_1_np[:, 1], ax[0], n_std = 2, edgecolor='red', linestyle='--')
        ax[0].set_title(f"Source_{model.__name__}")

        heatmap_model(self.x_2, self.y_2, ax[1], model)
        scatter_binary(self.x_2, self.y_2, ax[1])
        confidence_ellipse(x_2_np[:, 0], x_2_np[:, 1], ax[1], n_std = 2, edgecolor='red', linestyle='--')
        ax[1].set_title(f"Target_Acc={round(acc.item(), 2)}")

    def plot_all(self):
        if len(self.models) == 0:
            return 

        fig, axes = plt.subplots(2, len(self.models))
        desc = f"Bound_deg={self.boundary_degree}; mu_s={self.mu_s}; mu_t={self.mu_t}"
        fig.suptitle(desc)

        for i, model in enumerate(self.models):
            self._plot_model(axes[:, i], model)

        fig.tight_layout()

        fig.savefig(f'experiment_{desc}.png')            


