import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal
import scipy
import torch
from rba.util import get_poly_data

class BivariateGaussian:
    def __init__(self, mu_s, var_s, mu_t, var_t, num_points = 100):
        self.mu_s = mu_s
        self.var_s = var_s
        self.mu_t = mu_t
        self.var_t = var_t

        self.mvn_s = multivariate_normal(mu_s, var_s)
        self.mvn_t = multivariate_normal(mu_t, var_t)

        self.x_s = torch.FloatTensor(self.mvn_s.rvs(num_points))
        self.x_t = torch.FloatTensor(self.mvn_t.rvs(num_points))

        self.gen_rand_decision_boundary()

    def set_poly_features(self, poly_features):
        self.poly_features = poly_features
        self.x_s_poly = get_poly_data(self.x_s, poly_features)
        self.x_t_poly = get_poly_data(self.x_t, poly_features)
    
    def gen_rand_decision_boundary(self, boundary_degree = 3):
        samples = (self.mvn_s.rvs(boundary_degree + 1) 
            + self.mvn_t.rvs(boundary_degree + 1)) / 2
        x1 = samples[:, 0]
        x2 = samples[:, 1]

        A = np.array([x1**i for i in range(boundary_degree + 1)]).T
        A_inv = scipy.linalg.inv(A)
        params = np.dot(A_inv, x2)

        def get_y(x1, x2):
            temp = torch.ones((len(x1), len(samples))) # Tensor([x1**i for i in range(len(samples))])
            for i in range(len(samples)):
                temp[:, i] = x1**i
            y = torch.FloatTensor(np.dot(temp, params))
            return torch.where(x2 >= y, 1,0).reshape((-1, 1)).to(torch.float)
        
        self.get_y = get_y
        self.y_s = self.get_y(self.x_s[:, 0], self.x_s[:, 1])
        self.y_t = self.get_y(self.x_t[:, 0], self.x_t[:, 1])

    def gen_decision_boundary_points(self, samples : np.array):
        x1 = samples[:, 0]
        x2 = samples[:, 1]

        A = np.array([x1**i for i in range(len(samples))]).T
        A_inv = scipy.linalg.inv(A)
        params = np.dot(A_inv, x2)

        def get_y(x1, x2):
            temp = torch.ones((len(x1), len(samples))) # Tensor([x1**i for i in range(len(samples))])
            for i in range(len(samples)):
                temp[:, i] = x1**i
            y = torch.FloatTensor(np.dot(temp, params))
            return torch.where(x2 >= y, 1,0).reshape((-1, 1)).to(torch.float)
        
        self.get_y = get_y
        self.y_s = self.get_y(self.x_s[:, 0], self.x_s[:, 1])
        self.y_t = self.get_y(self.x_t[:, 0], self.x_t[:, 1])

if __name__ == "__main__":    
    mu_s = [6, 6] 
    var_s = [[3, -2], [-2, 3]] 
    mu_t = [7, 7] 
    var_t = [[3, 2], [2, 3]] 

    gaussian2 = BivariateGaussian(mu_s, var_s, mu_t, var_t)

    x_s, y_s, x_t, y_t = gaussian2.gen_data()

