import numpy as np
import torch
import torch.nn as nn

def rba_test(model, X_t, y_t):
    n_row, _ = X_t.shape
    
    model.eval()

    loss_fn = nn.BCELoss() 

    outputs = model(X_t)
    preds = outputs # torch.sigmoid(outputs * r_st)

    loss = loss_fn(preds.squeeze(), y_t.squeeze())
    acc = torch.sum(torch.round(preds) == y_t) / n_row
    print(f"Target Loss: {loss}. Target Accuracy: {acc}")
    return loss, preds, acc