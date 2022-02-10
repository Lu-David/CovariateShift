import numpy as np
from rba.train.log_train import log_train
from rba.test.log_test import log_test
import torch

def kernel_density_estimation():
    pass

def lr_density_estimation(X_s, X_t, weight_decays = [0.0625, 1, 16]):
    
    np.random.seed(10)

    ns_row, _ = X_s.shape
    nt_row, _ = X_t.shape

    inda_s = np.arange(ns_row)
    inda_t = np.arange(nt_row)

    nv_s = int(np.floor(0.2 * ns_row))
    nv_t = int(np.floor(0.2 * nt_row))

    indv_s = np.random.permutation(ns_row)[:nv_s] 
    indv_t = np.random.permutation(nt_row)[:nv_t]

    indt_s = np.setdiff1d(inda_s, indv_s)
    indt_t = np.setdiff1d(inda_t, indv_t)

    X_train = torch.cat((torch.FloatTensor(X_s[indt_s, :]), torch.FloatTensor(X_t[indt_t, :])))
    X_valid = torch.cat((torch.FloatTensor(X_s[indv_s, :]), torch.FloatTensor(X_t[indv_t, :])))
    
    y_train = torch.cat((torch.ones((ns_row - nv_s, 1)), torch.zeros((nt_row - nv_t, 1)) ))
    y_valid = torch.cat((torch.ones((nv_s, 1)), torch.zeros((nv_t, 1)) ))

    rt_st = torch.ones((ns_row + nt_row - nv_s - nv_t, 1))
    rv_st = torch.ones((nv_s + nv_t, 1))
    
    losses = torch.zeros((len(weight_decays), 1))
    for i, lamb in enumerate(weight_decays):
        model = log_train(X_train, y_train, rt_st, weight_decay=lamb)
        loss, pred, acc = log_test(model, X_valid, y_valid)
        losses[i] = loss

    ind_min = torch.argmin(loss)

    X_train = torch.cat((X_s, X_t))
    y_train = torch.cat((torch.ones((ns_row, 1)), torch.zeros((nt_row, 1))))
    r_st = torch.ones((ns_row + nt_row, 1))

    model = log_train(X_train, y_train, r_st, weight_decay=weight_decays[ind_min])
    _, pred, _ = log_test(model, X_train, y_train)

    d_ss = pred[:ns_row, 0]
    d_st = 1 - pred[:ns_row, 0]

    d_ts = pred[ns_row:, 0]
    d_tt = 1 - pred[ns_row:, 0]

    print("Finish Density Estimation")

    return d_ss, d_st, d_ts, d_tt, model