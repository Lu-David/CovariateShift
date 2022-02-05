function [logloss, prediction] = BinaryRobustTest(theta, X_t, y_t, r_st)
%testing algorithm for robust binary classification under covariate shift(using first order features)
%INPUT:
%       theta:      the theta parameter, a (n_col + 1) * 1 vector
%       X_t:        testing data, a n_row * n_col dimension matrix(without labels)
%       y_t:        testing labels, a n_row * 1 vector, in this binary
%                   case, either 1 or -1
%       r_st:        each test data's probability(density) under source distribution, over each 
%                   test data's probability(density) under target distribution, a n_row * 1 vector
%OUTPUT:
%       logloss:       logloss, log with base 2  (in binary, the worst case is logloss = 1)  
%       prediction:    P(y|x), each test data's probability of belonging to 1 or -1, a n_row * 2 vector



[n_row, ~] = size(X_t);
F = [ones(n_row, 1) X_t];
P = zeros(n_row, 1);
logloss = 0;
prediction = zeros(n_row ,2);

for i = 1 : n_row
    temp = r_st(i) * theta' * F(i, :)' * y_t(i);
    temp_max = max(temp, -temp);
    temp_min = min(temp, -temp);
    P(i) = exp(temp - temp_max - log(1 + exp(temp_min - temp_max)));
    logloss = logloss - log(P(i));
    if y_t(i) == 1
        prediction(i, 1) = P(i);
        prediction(i, 2) = 1 - P(i);
    else
        prediction(i, 2) = P(i);
        prediction(i, 1) = 1 - P(i);
    end
end
logloss = logloss / n_row / 0.6931;





