from matplotlib import markers
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import os


def two_dim_plot(model, dr_estimator, x, y):
    
    mean = torch.mean(x, axis=0)
    std = torch.std(x, axis = 0)
    
    maxs = mean + 4 * std
    mins = mean - 4 * std

    X_dim1, X_dim2 = np.meshgrid(np.arange(mins[0], maxs[0] + 0.1, 0.1), np.arange(mins[1], maxs[1] + 0.1, 0.1))
    dims = X_dim1.shape

    coors = np.dstack((X_dim1, X_dim2))
    coors = torch.FloatTensor(coors.reshape((dims[0] * dims[1], -1)))

    r_st = dr_estimator(coors)
    r_st = torch.clip(torch.Tensor(r_st).unsqueeze(1), -5000, 5000)

    predictions = model(coors, r_st)
    predictions = torch.reshape(predictions, (dims[0], dims[1]))

    plt.imshow(predictions.detach().numpy(), cmap='Spectral', interpolation='nearest', origin='lower', extent=[mins[0], maxs[0], mins[1], maxs[1]])

    pos = x[np.array(y == 1).flatten()]
    neg = x[np.array(y == 0).flatten()]
    plt.scatter(pos[:,0], pos[:,1], marker="x", color="black", s = 7)
    plt.scatter(neg[:,0], neg[:,1], marker="o", color="white", s = 7)

    description = f"{model.__name__}_{dr_estimator.__name__}"
    plt.title(description)
    dirname = os.path.dirname(__file__)
    foldername = os.path.join(dirname, "../figures/")
    plt.savefig(foldername + description + ".png")