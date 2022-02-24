import numpy as np
import torch
import torch.nn as nn

def log_test(model, X_t, y_t, r_st):
    n_row, _ = X_t.shape
    
    model.eval()

    loss_fn = nn.BCELoss() 

    outputs = model(X_t, r_st)
    loss = loss_fn(outputs.squeeze(), y_t.squeeze())
    acc = torch.sum(torch.round(outputs) == y_t) / n_row
    print(f"Target Loss: {loss}. Target Accuracy: {acc}")
    return loss, outputs, acc