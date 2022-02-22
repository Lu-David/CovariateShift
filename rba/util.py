from sklearn.preprocessing import PolynomialFeatures

def get_poly_data(x, poly_features):
    poly = PolynomialFeatures(poly_features, include_bias=False)
    return poly.fit_transform(x)