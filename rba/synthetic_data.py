import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal
import scipy
import torch
from sklearn.preprocessing import PolynomialFeatures

class BivariateGaussian:
    def __init__(self, mu_s, var_s, mu_t, var_t, boundary_degree = 3):
        self.mu_s = mu_s
        self.var_s = var_s
        self.mu_t = mu_t
        self.var_t = var_t

        self.mvn_s = multivariate_normal(mu_s, var_s)
        self.mvn_t = multivariate_normal(mu_t, var_t)

        self.boundary_degree = boundary_degree
    
    def gen_rand_decision_boundary(self):
        samples = (self.mvn_s.rvs(self.boundary_degree + 1) 
            + self.mvn_t.rvs(self.boundary_degree + 1)) / 2
        x1 = samples[:, 0]
        x2 = samples[:, 1]

        A = np.array([x1**i for i in range(self.boundary_degree + 1)]).T
        A_inv = scipy.linalg.inv(A)
        params = np.dot(A_inv, x2)

        def get_y(x1, x2):
            temp = np.array([x1**i for i in range(self.boundary_degree + 1)])
            y = np.dot(params, temp)
            return np.where(x2 >= y, 1,0).reshape((-1, 1))
        
        return get_y

    def gen_data(self, num_points = 100, noise = 0.05):
        # TODO add noise
        x_s = self.mvn_s.rvs(num_points)
        x_t = self.mvn_t.rvs(num_points)

        get_y = self.gen_rand_decision_boundary()
        
        y_s = get_y(x_s[:, 0], x_s[:, 1])
        y_t = get_y(x_t[:, 0], x_t[:, 1])

        return torch.Tensor(x_s), torch.Tensor(y_s), torch.Tensor(x_t), torch.Tensor(y_t)


if __name__ == "__main__":    
    mu_s = [6, 6] 
    var_s = [[3, -2], [-2, 3]] 
    mu_t = [7, 7] 
    var_t = [[3, 2], [2, 3]] 

    gaussian2 = BivariateGaussian(mu_s, var_s, mu_t, var_t)

    x_s, y_s, x_t, y_t = gaussian2.gen_data()

