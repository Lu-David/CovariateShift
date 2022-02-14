import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class LogClassifier(nn.Module):
    """
    Single layer classifier that supports logistic regression and importance reweighting methods
    """

    def __init__(self, dr_estimator, in_features = 2, out_features = 1, bias = True):
        """[summary]

        Args:
            in_features (int, optional): number of columns in X. Defaults to 2.
            out_features (int, optional): number of classes in Y. Defaults to 1.
            bias (bool, optional): Defaults to False.
        """

        super(LogClassifier, self).__init__()
        
        self.layer1 = nn.Linear(in_features, out_features, bias = bias)
        self.activation1 = nn.Sigmoid()
        self.dr_estimator = dr_estimator
        self.__name__ = "Log_" + dr_estimator.__name__

    def forward(self, input):
        """[summary]

        Args:
            input (torch.Tensor): features

        Returns:
            torch.Tensor: predictions
        """
        r_st = self.dr_estimator(input).detach()
        r_st = torch.clip(r_st, -5000, 5000)
        return self.activation1(self.layer1(input) * r_st)
    