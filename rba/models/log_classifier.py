import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class LogClassifier(nn.Module):
    """
    Single layer classifier that supports logistic regression and importance reweighting methods
    """

    def __init__(self,in_features = 2, out_features = 1, bias = True):
        """[summary]
        Args:
            in_features (int, optional): number of columns in X. Defaults to 2.
            out_features (int, optional): number of classes in Y. Defaults to 1.
            bias (bool, optional): Defaults to False.
        """

        super(LogClassifier, self).__init__()
        
        self.layer1 = nn.Linear(in_features, out_features, bias = bias)
        self.activation1 = nn.Sigmoid()

    def forward(self, input, r_ts):
        r_ts = r_ts.detach()
        """[summary]
        Args:
            input (torch.Tensor): features
        Returns:
            torch.Tensor: predictions
        """
        return self.activation1(self.layer1(input) * r_ts)