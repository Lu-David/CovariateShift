function [ logloss, P ] = MultiClassRobustTest(theta, X_t, y_t, n_class, r_st)
%testing algorithm for robust multiclass classification under covariate shift(using first order features)
%INPUT:
%       theta:      the theta parameter, a (n_col + 1) * 1 vector
%       X_t:        testing data, a n_row * n_d dimension matrix(without labels)
%       y_t:        testing labels, a n_row * 1 vector, in this binary
%                   case, range from 1 to N, continuously
%       n_class:    number of class N, a scalar
%       r_st:       each test data's probability(density) under source distribution, over each 
%                   test data's probability(density) under target distribution, a n_row * 1 vector
%OUTPUT:
%       logloss:       logloss, log with base 2  (the worst case is  -log(1/N))  
%       prediction:    P(y|x), each test data's probability of belonging to each class, a n_row * n_class vector



[n_row, ~] = size(X_t);
F = X_t;
P = zeros(n_row, n_class);
logloss = 0;

for i = 1 : n_row
    exp_temp = theta * F(i, :)' * r_st(i);
    exp_temp = exp_temp - max(exp_temp); %avoiding overflow or underflow
    sum_exp = sum(exp(exp_temp));
    P(i, :) = exp(exp_temp - log(sum_exp));
    logloss = logloss - log(P(i, y_t(i)));    
end
logloss = logloss / n_row / 0.6931;


end

