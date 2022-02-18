import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import torch

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    # if x.size != y.size:
    #     raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def scatter_binary(x, y, ax):
    pos = x[np.array(y == 1).flatten()]
    neg = x[np.array(y == 0).flatten()]
    ax.scatter(pos[:,0], pos[:,1], marker="x", color="black", s = 7)
    ax.scatter(neg[:,0], neg[:,1], marker="o", color="white", s = 7)

def heatmap_model(x, y, ax, model):
    mean = torch.mean(x, axis=0)
    std = torch.std(x, axis=0)
    
    maxs = torch.max(x, dim = 0).values
    mins = torch.min(x, dim = 0).values

    res = 0.01
    X_dim1, X_dim2 = np.meshgrid(np.arange(mins[0], maxs[0] + res, res), np.arange(mins[1], maxs[1] + res, res))
    dims = X_dim1.shape

    coors = np.dstack((X_dim1, X_dim2))
    coors = torch.FloatTensor(coors.reshape((dims[0] * dims[1], -1)))

    model.eval()
    predictions = model(coors)
    predictions = torch.reshape(predictions, (dims[0], dims[1]))

    ax.imshow(predictions.detach().numpy(), cmap='Spectral', interpolation='nearest', origin='lower', extent=[mins[0], maxs[0], mins[1], maxs[1]])