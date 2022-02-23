from pyexpat import model
import numpy as np
from rba.train.log_train import log_train
from rba.test.log_test import log_test
from rba.util import get_poly_data
import torch
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


def ones(x):
    return torch.ones((x.shape[0], 1))

def get_kernel_density_estimator(X_s, X_t, bandwidth=0.7):
    # Note: Changing bandwidth is an important parameter to tune!
    kde_s = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X_s)
    kde_t = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X_t)

    def kde(x):
        return torch.Tensor(np.exp(kde_s.score_samples(x)) / np.exp(kde_t.score_samples(x))).unsqueeze(1)

    return kde

def get_mvn_estimator(mu_s, var_s, mu_t, var_t):
    mvn_s = multivariate_normal(mu_s, var_s)
    mvn_t = multivariate_normal(mu_t, var_t)
    
    def mvn(x):
        return torch.Tensor(mvn_s.pdf(x) / mvn_t.pdf(x)).unsqueeze(1)

    return mvn

# TODO fix gmm
def get_gmm_estimator(X_s, X_t):
    _, n_col = X_s.shape
    n_components = n_col
    means_init = torch.stack((
        torch.mean(X_s, dim=0),
        torch.mean(X_t, dim=0),
    ), dim=0)

    print("means_init", means_init)

    X_s = torch.Tensor(X_s)
    X_t = torch.Tensor(X_t)

    ns_row, _ = X_s.shape
    nt_row, _ = X_t.shape

    inda_s = np.arange(ns_row)
    inda_t = np.arange(nt_row)

    nv_s = int(np.floor(0. * ns_row))
    nv_t = int(np.floor(0. * nt_row))

    indv_s = np.random.permutation(ns_row)[:nv_s] 
    indv_t = np.random.permutation(nt_row)[:nv_t]

    indt_s = np.setdiff1d(inda_s, indv_s)
    indt_t = np.setdiff1d(inda_t, indv_t)

    X_train = torch.cat((torch.FloatTensor(X_s[indt_s, :]), torch.FloatTensor(X_t[indt_t, :])))
    X_valid = torch.cat((torch.FloatTensor(X_s[indv_s, :]), torch.FloatTensor(X_t[indv_t, :])))
    
    y_train = torch.cat((torch.ones((ns_row - nv_s, 1)), torch.zeros((nt_row - nv_t, 1)) ))
    y_valid = torch.cat((torch.ones((nv_s, 1)), torch.zeros((nv_t, 1)) ))
    
    model = GaussianMixture(n_components, means_init=means_init).fit(X_train, y_train)

    def gmm(x):
        pred = model.predict(x)
        pred[pred == 0] = 0.001
        return torch.Tensor(pred / (1 - pred)).unsqueeze(1)
    
    return gmm

def get_lrdr_estimator(X_s, X_t, weight_decays = [1, 5, 15]):

    poly_features = 2

    # X_s = np.concatenate((
    #     X_s[:, [0]], X_s[:, [1]], X_s[:, [0]] ** 2, X_s[:, [0]] * X_s[:, [1]], X_s[:, [1]] ** 2
    # ), axis=1)

    # X_t = np.concatenate((
    #     X_t[:, [0]], X_t[:, [1]], X_t[:, [0]] ** 2, X_t[:, [0]] * X_t[:, [1]], X_t[:, [1]] ** 2
    # ), axis=1)

    X_s = get_poly_data(X_s, poly_features)
    X_t = get_poly_data(X_t, poly_features)

    X_s = torch.Tensor(X_s)
    X_t = torch.Tensor(X_t)

    ns_row, _ = X_s.shape
    nt_row, _ = X_t.shape

    inda_s = np.arange(ns_row)
    inda_t = np.arange(nt_row)

    nv_s = int(np.floor(0.1 * ns_row))
    nv_t = int(np.floor(0.1 * nt_row))

    indv_s = np.random.permutation(ns_row)[:nv_s] 
    indv_t = np.random.permutation(nt_row)[:nv_t]

    indt_s = np.setdiff1d(inda_s, indv_s)
    indt_t = np.setdiff1d(inda_t, indv_t)

    X_train = torch.cat((torch.FloatTensor(X_s[indt_s, :]), torch.FloatTensor(X_t[indt_t, :])))
    X_valid = torch.cat((torch.FloatTensor(X_s[indv_s, :]), torch.FloatTensor(X_t[indv_t, :])))
    
    y_train = torch.cat((torch.ones((ns_row - nv_s, 1)), torch.zeros((nt_row - nv_t, 1)) ))
    y_valid = torch.cat((torch.ones((nv_s, 1)), torch.zeros((nv_t, 1)) ))
    
    losses = torch.zeros((len(weight_decays), 1))
    for i, lamb in enumerate(weight_decays):
        model = log_train(X_train, y_train, ones, weight_decay=lamb)
        loss, pred, acc = log_test(model, X_valid, y_valid)
        losses[i] = loss
    ind_min = torch.argmin(loss)

    X_train = torch.cat((X_s, X_t))
    y_train = torch.cat((torch.ones((ns_row, 1)), torch.zeros((nt_row, 1))))

    model = log_train(X_train, y_train, ones, max_itr=10000, weight_decay=weight_decays[ind_min])

    def lrdr(x):
        x = torch.Tensor(get_poly_data(x, poly_features))
        pred = model(x)
        return pred / (1 - pred)

    return lrdr