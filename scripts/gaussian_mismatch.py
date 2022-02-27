from rba.experiment import BivariateExperiment
from rba.synthetic_data import BivariateGaussian
from rba.train.rba_train import rba_train
from rba.density_estimation import get_mvn_estimator
from rba.util import get_poly_data

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

x_1 = torch.FloatTensor(x_1)
x_2 = torch.FloatTensor(x_2)
y_1 = torch.FloatTensor(np.where(y_1 == 1, 1, 0))
y_2 = torch.FloatTensor(np.where(y_2 == 1, 1, 0))

mu_s = [6, 6] 
var_s = [[3, -2], [-2, 3]] 
mu_t = [7, 7] 
var_t = [[3, 2], [2, 3]] 

F = [1, 2, 3]
B = [
    np.array([ # Linear 
            [5, 5],
            [6, 7.5],
    ]), 
    np.array([ # Quadratic 
            [5, 5],
            [6, 7.5],
            [10, 2.5]
    ]),  
    np.array([ # Cubic
            [2.5, 2.5],
            [5, 10],
            [7.5, 0],
            [10, 15]
    ])
]

gaussian = BivariateGaussian(mu_s, var_s, mu_t, var_t)
gaussian.x_s = x_1
gaussian.y_s = y_1
gaussian.x_t = x_2
gaussian.y_t = y_2
for f in F:
    gaussian.set_poly_features(f)
    for b in B:
        gaussian.gen_decision_boundary_points(b) 
        experiment = BivariateExperiment(gaussian)

        # experiment.title = f"mvn_gaussian1_B{len(b) - 1}_F{f}"
        # experiment.set_dr_estimator("mvn")
        # experiment.train_all()
        # experiment.plot_all()

        # experiment.title = f"kde_gaussian1_B{len(b) - 1}_F{f}"
        # experiment.set_dr_estimator("kde")
        # experiment.train_all()
        # experiment.plot_all()

        # experiment.title = f"lrdr_gaussian1_B{len(b) - 1}_F{f}"
        # experiment.set_dr_estimator("lrdr")
        # experiment.train_all()
        # experiment.plot_all()

        experiment.title = f"gmm_gaussian1_B{len(b) - 1}_F{f}"
        experiment.set_dr_estimator("gmm")
        experiment.train_all()
        experiment.plot_all()


