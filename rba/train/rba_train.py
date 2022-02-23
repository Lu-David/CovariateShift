import torch
import torch.nn as nn
import numpy as np
import torchviz
from rba.models.rba_classifier import RBAClassifier
from rba.util import get_poly_data

def rba_train(X_s, y_s, dr_estimator, max_itr = 10000, lr = 0.01, weight_decay = 0, poly_features = 1):

    X_s = get_poly_data(X_s, poly_features)

    _, n_col = X_s.shape
    _, out_features = y_s.shape

    model = RBAClassifier(dr_estimator=dr_estimator, in_features = n_col, out_features=out_features)
    loss_fn = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    for param in model.parameters():
        param.data = nn.parameter.Parameter(torch.ones_like(param))

    model.train()

    for i in range(max_itr): 
        optimizer.zero_grad()
        
        outputs = model.forward(X_s)
        # print(torchviz.make_dot(outputs.mean(), params=dict(model.named_parameters())))
        outputs.backward(y_s)
        optimizer.step()
        
        loss = loss_fn(outputs.squeeze(), y_s.squeeze())
        if i % 1000 == 0:
            print(f"Loss at step {i}: {float(loss.data)}")

    return model

    

if __name__ == "__main__":
    pass