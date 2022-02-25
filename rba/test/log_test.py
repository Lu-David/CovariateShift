import numpy as np
import torch
import torch.nn as nn
from rba.util import get_poly_data

def log_test(model, X_t, y_t, r_st):
    n_row, _ = X_t.shape
    
    model.eval()

    loss_fn = nn.BCELoss() 
    F = torch.cat((
        torch.ones(r_st.shape), 
        X_t
        ), dim = 1)
    outputs = model(F, r_st)
    loss = loss_fn(outputs.squeeze(), y_t.squeeze())
    acc = torch.sum(torch.round(outputs) == y_t) / n_row
    print(f"Target Loss: {loss}. Target Accuracy: {acc}")
    return loss, outputs, acc