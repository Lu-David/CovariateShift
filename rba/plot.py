from matplotlib import markers
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch


def two_dim_plot(model, x,):
    
    maxs = 15
    mins = -5

    X_dim1, X_dim2 = np.meshgrid(np.arange(mins, maxs + 0.1, 0.1), np.arange(mins, maxs + 0.1, 0.1))

    dim = int((maxs - mins) / 0.1 + 1)

    prediction = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(dim):
            x_t = torch.FloatTensor([X_dim1[i, j], X_dim2[i, j]])
            outputs = rba_model(x_t, mvn_s.pdf(x_t) / mvn_t.pdf(x_t))
            prediction[i, j] = outputs[0]

    plt.imshow(prediction, cmap='Spectral', interpolation='nearest', origin='lower', extent=[-5, 15, -5, 15])

    pos = x_1[np.array(y_1 == 1).flatten()]
    neg = x_1[np.array(y_1 == -1).flatten()]

    plt.scatter(pos[:,0], pos[:,1], marker="x", color="black", s = 7)
    plt.scatter(neg[:,0], neg[:,1], marker="o", color="white", s = 7)
    plt.title("RBA - First Order Features")