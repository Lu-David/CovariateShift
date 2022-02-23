from rba.train.rba_train import rba_train
from rba.train.log_train import log_train
from rba.test.rba_test import rba_test
from rba.test.log_test import log_test
from rba.density_estimation import get_kernel_density_estimator, get_mvn_estimator, get_lrdr_estimator, ones, get_gmm_estimator
from rba.plot import heatmap_model, scatter_binary, confidence_ellipse
from rba.synthetic_data import BivariateGaussian
from rba.util import get_poly_data


import scipy.io
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

class BivariateExperiment():

    def __init__(self, mu_s, var_s, mu_t, var_t, poly_features = 1):
        self.mu_s = mu_s
        self.var_s = var_s
        self.mu_t = mu_t 
        self.var_t = var_t
        self.poly_features = poly_features
        self.boundary_degree = 1

        gaussian = BivariateGaussian(mu_s, var_s, mu_t, var_t, self.boundary_degree)
        self.x_1, self.y_1, self.x_2, self.y_2 = gaussian.gen_data()

        self.x_1 = get_poly_data(self.x_1, self.poly_features)
        self.x_2 = get_poly_data(self.x_2, self.poly_features)

        self.kde_dr = get_kernel_density_estimator(self.x_1, self.x_2)
        self.mvn_dr = get_mvn_estimator(mu_s, var_s, mu_t, var_t)
        # self.gmm_dr = get_gmm_estimator(self.x_1, self.x_2)
        self.lr_dr = get_lrdr_estimator(self.x_1, self.x_2)
        self.ones_dr = ones
        
        self.models = []

    def train_all(self, dr_estimator):
        rba_model = rba_train(self.x_1, self.y_1, dr_estimator, max_itr = 10000, lr = 0.01, poly_features=1) 
        iw_model = log_train(self.x_1, self.y_1, dr_estimator, max_itr = 10000, lr = 0.01, poly_features=1) 
        log_model = log_train(self.x_1, self.y_1, self.ones_dr, max_itr = 10000, lr = 0.01, poly_features=1) 
        self.models = [rba_model, iw_model, log_model]

    def test_all(self):
        if len(self.models) == 0:
            return 
        for model in self.models:
            loss, preds, acc = log_test(model, self.x_2, self.y_2)

    def _plot_model(self, ax, model):

        log, preds, acc_1 = log_test(model, self.x_1, self.y_1)

        log, preds, acc_2 = log_test(model, self.x_2, self.y_2) # log_test and rba_test are the same

        x_1_np = self.x_1.detach().numpy()
        x_2_np = self.x_2.detach().numpy()

        heatmap_model(self.x_1, self.y_1, ax[0], model, poly_features=self.poly_features)
        scatter_binary(self.x_1, self.y_1, ax[0])
        confidence_ellipse(x_1_np[:, 0], x_1_np[:, 1], ax[0], n_std = 2, edgecolor='red', linestyle='--')
        ax[0].set_title(f"Source_{model.__name__}_Acc={round(acc_1.item(), 2)}")

        heatmap_model(self.x_2, self.y_2, ax[1], model, poly_features=self.poly_features)
        scatter_binary(self.x_2, self.y_2, ax[1])
        confidence_ellipse(x_2_np[:, 0], x_2_np[:, 1], ax[1], n_std = 2, edgecolor='red', linestyle='--')
        ax[1].set_title(f"Target_Acc={round(acc_2.item(), 2)}")

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


