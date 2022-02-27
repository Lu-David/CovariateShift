from sklearn.preprocessing import PolynomialFeatures
import torch

def get_poly_data(x, poly_features):
    poly = PolynomialFeatures(poly_features, include_bias=False)
    return torch.FloatTensor(poly.fit_transform(x))