import numpy as np
import torch
import torch.nn as nn

def rba_test(model, X_t, y_t, r_st):
    """[summary]

    Args:
        model ([type]): [description]
        X_t ([type]): [description]
        y_t ([type]): [description]
        r_st ([type]): [description]

    Returns:
        [type]: [description]
    """
    n_row, _ = X_t.shape
    
    model.eval()

    loss_fn = nn.BCELoss() 

    outputs = model(X_t, r_st)
    preds = outputs # torch.sigmoid(outputs * r_st)

    loss = loss_fn(preds.squeeze(), y_t.squeeze())
    acc = torch.sum(torch.round(preds) == y_t) / n_row
    return loss, preds, acc