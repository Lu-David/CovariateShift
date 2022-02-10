import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal

class Synthesizer:
    def __init__(self):
        pass


if __name__ == "__main__":    

    # decision boundary 
    def decision(x1, x2):
        if x2 > 12 - x1:
            return 1
        return 0

    def noisy_decision(decision):
        if np.random.random() < 0.07:
            return int(not decision)
        return decision


    # source mean 
    mu_s = [6, 6] 

    # source variance
    var_s = [[3, -2], [-2, 3]] 

    # target mean
    mu_t = [7, 7] 

    # target variance
    var_t = [[3, 2], [2, 3]] 

    mvn_s = multivariate_normal(mu_s, var_s)
    mvn_t = multivariate_normal(mu_t, var_t)

    num = 100
    x = np.zeros((num, 2))
    y = np.zeros((num, 1))
    for i in range(num):
        x[i] = mvn_s.rvs()
        y[i] = noisy_decision(decision(*x[i]))

    pos = x[np.array(y == 1).flatten()]
    neg = x[np.array(y == 0).flatten()]

    plt.scatter(pos[:, 0], pos[:, 1], marker = "x", color="black", s = 11)
    plt.scatter(neg[:, 0], neg[:, 1], marker = "o", color="blue", s = 11)
    plt.show()
    plt.savefig("./figure")
    