import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class RBAGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, r_st):
        pred = torch.sigmoid(output * r_st)
        ctx.save_for_backward(pred)
        return pred

    @staticmethod
    def backward(ctx, y):
        prob, = ctx.saved_tensors
        grad_temp = y.clone()
        grad_input = (prob - grad_temp)
        return grad_input, None

class RBAClassifier(nn.Module):
    """
    Single layer robust bias aware (RBA) classifier
    """

    def __init__(self, dr_estimator, in_features = 2, out_features = 1, degree = 1, bias = True):
        """[summary]

        Args:
            in_features (int, optional): number of columns in X. Defaults to 2.
            out_features (int, optional): number of classes in Y. Defaults to 1.
            bias (bool, optional): Defaults to False.
        """

        super(RBAClassifier, self).__init__()
        self.layer1 = nn.Linear(in_features, out_features, bias = bias)
        self.__name__ = "RBA_" + dr_estimator.__name__
        self.dr_estimator = dr_estimator
        self.r_st = None
        self.prev = None

    def forward(self, input):
        """[summary]

        Args:
            input (torch.Tensor): features

        Returns:
            torch.Tensor: predictions
        """
        # TODO: Fix the hacky workaround to limit recalcultion of density ratios
        if self.r_st == None or len(input) != self.prev:
            self.r_st = self.dr_estimator(input).detach()
            self.r_st = torch.clip(self.r_st, -5000, 5000)
            self.prev = len(input)

        return RBAGrad.apply(self.layer1(input), self.r_st)