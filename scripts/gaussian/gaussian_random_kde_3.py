from sklearn.mixture import GaussianMixture
from rba.experiment import BivariateExperiment

mu_s = [8, 11] 
var_s = [[9, -5], [-5, 9]] 
mu_t = [10, 10] 
var_t = [[7, 4], [4, 7]] 

experiment = BivariateExperiment(mu_s, var_s, mu_t, var_t)
experiment.train_all(experiment.kde_dr)
experiment.plot_all()