from rba.train.rba_train import rba_train
from rba.test.rba_test import rba_test
from rba.test.log_test import log_test
from rba.density_estimation import get_kernel_density_estimator, get_mvn_estimator, get_lrdr_estimator, ones, get_gmm_estimator, inverse
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

    def __init__(self, gaussian : BivariateGaussian):

        self.gaussian = gaussian

        self.dr_estimator_ls = [
            get_mvn_estimator(self.gaussian.mu_s, self.gaussian.var_s, self.gaussian.mu_t, self.gaussian.var_t),
            get_kernel_density_estimator(self.gaussian.x_s, self.gaussian.x_t),
            get_lrdr_estimator(self.gaussian.x_s, self.gaussian.x_t),
            get_gmm_estimator(self.gaussian.x_s, self.gaussian.x_t),
        ]

        self.dr_estimator = None
        
        self.models = []
        
        self.title = "No Title"

    def set_dr_estimator(self, name):
        names = []
        for estimator in self.dr_estimator_ls:
            names.append(estimator.__name__)
            if estimator.__name__ == name:
                self.dr_estimator = estimator
                return
        raise Exception(f"{name} density estimator does not exist! Available estimators are: {names}")

    def train_all(self):
        r_st = self.dr_estimator(self.gaussian.x_s)
        r_ts = 1 / r_st
        ones = torch.ones(r_st.shape)

        rba_model = rba_train(self.gaussian.x_s_poly, self.gaussian.y_s, r_st, ones) 
        iw_model = rba_train(self.gaussian.x_s_poly, self.gaussian.y_s, ones, r_ts)
        log_model = rba_train(self.gaussian.x_s_poly, self.gaussian.y_s, ones, ones)
        self.models = [rba_model, iw_model, log_model]

    def test_all(self):
        if len(self.models) == 0:
            return 
        for model in self.models:
            loss, preds, acc = log_test(model, self.gaussian.x_t, self.gaussian.y_t)

    def _plot_model(self, ax, model, dr_estimator):
        
        r_st_1 = self.dr_estimator(self.gaussian.x_s)
        r_st_2 = self.dr_estimator(self.gaussian.x_t)

        log, preds, acc_1 = log_test(model, self.gaussian.x_s_poly, self.gaussian.y_s, r_st_1)
        log, preds, acc_2 = log_test(model, self.gaussian.x_t_poly, self.gaussian.y_t, r_st_2) # log_test and rba_test are the same

        x_1_np = self.gaussian.x_s.detach().numpy()
        x_2_np = self.gaussian.x_t.detach().numpy()

        heatmap_model(self.gaussian.x_s, ax[0], model, dr_estimator, poly_features=self.gaussian.poly_features)
        scatter_binary(self.gaussian.x_s, self.gaussian.y_s, ax[0])
        confidence_ellipse(x_1_np[:, 0], x_1_np[:, 1], ax[0], n_std = 2, edgecolor='red', linestyle='--')
        ax[0].set_title(f"Source_Acc={round(acc_1.item(), 2)}")

        heatmap_model(self.gaussian.x_t, ax[1], model, dr_estimator, poly_features=self.gaussian.poly_features)
        scatter_binary(self.gaussian.x_t, self.gaussian.y_t, ax[1])
        confidence_ellipse(x_2_np[:, 0], x_2_np[:, 1], ax[1], n_std = 2, edgecolor='red', linestyle='--')
        ax[1].set_title(f"Target_Acc={round(acc_2.item(), 2)}")

    def plot_all(self):
        if len(self.models) == 0:
            return 

        fig, axes = plt.subplots(2, len(self.models))
        desc = self.title
        fig.suptitle(desc)
        
        self._plot_model(axes[:, 0], self.models[0], self.dr_estimator)
        self._plot_model(axes[:, 1], self.models[1], ones)
        self._plot_model(axes[:, 2], self.models[2], ones)

        fig.tight_layout()

        fig.savefig(f'{self.title}.png')            


