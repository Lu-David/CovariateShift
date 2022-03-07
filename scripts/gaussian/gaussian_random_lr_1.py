from sklearn.mixture import GaussianMixture
from rba.experiment import BivariateExperiment

mu_s = [6, 6] 
var_s = [[3, -2], [-2, 3]] 
mu_t = [7, 7] 
var_t = [[3, 2], [2, 3]] 

experiment = BivariateExperiment(mu_s, var_s, mu_t, var_t, poly_features=1)
experiment.train_all(experiment.lr_dr)
experiment.plot_all()