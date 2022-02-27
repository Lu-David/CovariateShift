import unittest

import torch  
from rba.synthetic_data import BivariateGaussian

mu_s = [6, 6] 
var_s = [[3, -2], [-2, 3]] 
mu_t = [7, 7] 
var_t = [[3, 2], [2, 3]] 

class SyntheticDataTestCase(unittest.TestCase):   

    def setUp(self) -> None:
        gaussian = BivariateGaussian(mu_s, var_s, mu_t, var_t, self.boundary_degree)
        x_1, y_1, x_2, y_2 = gaussian.gen_data()

 
if __name__ == '__main__':
    unittest.main()