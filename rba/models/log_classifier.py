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
        self.__name__ = "Log_" + dr_estimator.__name__
        self.dr_estimator = dr_estimator
        self.r_ts = None
        self.prev = None

    def forward(self, input):
        """[summary]

        Args:
            input (torch.Tensor): features

        Returns:
            torch.Tensor: predictions
        """

        if self.r_ts == None or len(input) != self.prev:
            self.r_ts = 1 / self.dr_estimator(input).detach()
            self.r_ts = torch.clip(self.r_ts, -5000, 5000)
            self.prev = len(input)
        
        if self.training:
            return self.activation1(self.layer1(input) * self.r_ts)

        return self.activation1(self.layer1(input))
    