import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import PolynomialFeatures

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
    def __init__(self, in_features = 2, out_features = 1, bias = True, poly_features = 1):
        """[summary]

        Args:
            in_features (int, optional): number of columns in X. Defaults to 2.
            out_features (int, optional): number of classes in Y. Defaults to 1.
            bias (bool, optional): Defaults to False.
        """

        # self.poly = PolynomialFeatures(poly_features, include_bias = False)
        super(RBAClassifier, self).__init__()
        # temp = torch.ones((1, in_features))
        # self.poly_temp = self.poly.fit_transform(temp)
        # self.in_features = self.poly_temp.shape[1]

        self.layer1 = nn.Linear(in_features, out_features, bias = bias)

    def forward(self, input, r_st):
        """[summary]

        Args:
            input (torch.Tensor): features

        Returns:
            torch.Tensor: predictions
        """
        # poly_input = torch.Tensor(self.poly.fit_transform(input))
        return RBAGrad.apply(self.layer1(input), r_st)