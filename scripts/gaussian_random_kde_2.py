from sklearn.mixture import GaussianMixture
from rba.experiment import BivariateExperiment

mu_s = [1, 1] 
var_s = [[3, -2], [-2, 3]] 
mu_t = [10, 10] 
var_t = [[3, 2], [2, 3]] 

experiment = BivariateExperiment(mu_s, var_s, mu_t, var_t)
experiment.train_all(experiment.kernel_dr)
experiment.plot_all()