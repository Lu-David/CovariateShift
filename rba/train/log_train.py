import torch
import torch.nn as nn
import numpy as np
import torchviz
from rba.models.log_classifier import LogClassifier

def log_train(X_s, y_s, dr_estimator, max_itr = 10000, lr = 0.01, weight_decay = 0):

  _, n_col = X_s.shape
  _, out_features = y_s.shape

  lr_model = LogClassifier(dr_estimator, in_features = n_col, out_features=out_features)
  loss_fn = nn.BCELoss() 
  optimizer = torch.optim.Adam(lr_model.parameters(), lr = lr, weight_decay = weight_decay)
  for param in lr_model.parameters():
    param.data = nn.parameter.Parameter(torch.ones_like(param))

  lr_model.train()
  
  for i in range(max_itr): 
      optimizer.zero_grad()
      
      outputs = lr_model(X_s)
      loss = loss_fn(outputs.squeeze(), y_s.squeeze())

      loss.backward()

      optimizer.step()

      if i % 1000 == 0:
        print(f"Loss at step {i}: {float(loss.data)}")

  return lr_model