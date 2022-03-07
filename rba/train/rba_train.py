import torch
import torch.nn as nn
import numpy as np
import torchviz
from rba.models.rba_classifier import RBAClassifierSimple

def rba_train(X_s, y_s, r_st, r_ts, max_itr = 10000, lr = 0.01, weight_decay = 0):

    _, n_col = X_s.shape
    _, out_features = y_s.shape

    bias = True
    if True: # not torch.equal(r_ts, torch.ones(r_ts.shape)):
        bias = False
        X = torch.cat((
            torch.ones(r_ts.shape), 
            X_s
        ), dim = 1)
    
    model = RBAClassifierSimple(in_features = n_col + int(not bias), out_features=out_features, bias = bias)
        
    loss_fn = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    for param in model.parameters():
        param.data = nn.parameter.Parameter(torch.ones_like(param))

    F = X * r_ts.detach()

    model.train()

    for i in range(max_itr): 
        optimizer.zero_grad()
        
        outputs = model.forward(F, r_st)
        outputs.backward(y_s)
        optimizer.step()
        
        loss = loss_fn(outputs.squeeze(), y_s.squeeze())
        if i % 1000 == 0:
            print(f"Loss at step {i}: {float(loss.data)}")

    for param in model.parameters():
        print(param.data)
    return model

if __name__ == "__main__":
    pass