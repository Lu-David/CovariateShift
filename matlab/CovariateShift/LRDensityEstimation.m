function [ d_ss, d_st, d_ts, d_tt ] = LRDensityEstimation( X_s, X_t, varargin)
%Density Estimation using logistic regression
%LR training and testing will be using BinaryRobustTrain and
%BinaryRobustTest, but with r_st set to all ones.
%
%INPUT:     
%           X_s:        source data, a ns_row * n_col matrix
%           X_t:        target data, a nt_row * n_col matrix
%           varargin:   
%                       'lambdas' => the range of lambdas, a m * 1 vector
%OUTPUT:    
%           d_ss:       density of each source data in source distribution, a ns_row * 1 vector 
%           d_st:       density of each source data in target distribution, a ns_row * 1 vector
%           d_ts:       density of each target data in source distribution, a nt_row * 1 vector
%           d_tt:       density of each target data in target distribution, a nt_row * 1 vector

disp('Starting Density Estimation....')
%set range of lambda
lambdas = [2^(-4) 1 2^4];


for i = 1 : 2 : length(varargin)
    name = varargin{i};
    value = varargin{i+1};
    switch name
        case 'lambdas'
            lambdas = value;
        otherwise;
    end
end



%determining the lambda that used to do the density estimation
%get size of source and target data
[ns_row, ~] = size(X_s);
[nt_row, ~] = size(X_t);


%sample validation set
inda_s = 1 : 1 : ns_row;
inda_t = 1 : 1 : nt_row;
nv_s = floor(0.2 * ns_row);
nv_t = floor(0.2 * nt_row);
indv_s = randperm(ns_row, nv_s);
indv_t = randperm(nt_row, nv_t);
indt_s = setdiff(inda_s, indv_s);
indt_t = setdiff(inda_t, indv_t);




X_train = [X_s(indt_s, :); X_t(indt_t, :)];
X_valid = [X_s(indv_s, :); X_t(indv_t, :)];

%construct y's and ratios
y_train = [ones(ns_row - nv_s, 1); -1 * ones(nt_row - nv_t, 1)];
y_valid = [ones(nv_s, 1); -1 * ones(nv_t, 1)];
rt_st = ones(ns_row + nt_row - nv_s - nv_t, 1);
rv_st = ones(nv_s + nv_t, 1);


%start validation
logloss = zeros(size(lambdas, 2), 1);
for i = 1 : size(lambdas, 2)
    theta = BinaryRobustTrain(X_train, y_train, rt_st, rt_st, 'lambda', lambdas(1, i), 'min_graident', 0.1);
    [~, pred] = BinaryRobustTest(theta, X_valid, y_valid, rv_st);
    logloss(i) = (-sum(log(pred(1 : nv_s, 1))) - sum(log(pred(nv_s + 1 : nv_s + nv_t, 2))))/(nv_s + nv_t)/0.6931;
end

[ ~ ,ind_min] = min(logloss);

%reconstruct data 
X_train = [ X_s; X_t ];
y_train = [ ones(ns_row, 1); -1 * ones(nt_row, 1) ];
r_st = ones(ns_row + nt_row, 1);


%train and testing
theta = BinaryRobustTrain(X_train, y_train, r_st, r_st, 'lambda', lambdas(1, ind_min));
[~, pred] = BinaryRobustTest(theta, X_train, y_train, r_st);

%get ratios out of pred
d_ss = pred(1 : ns_row, 1);
d_st = pred(1 : ns_row, 2);
d_ts = pred(ns_row + 1 : ns_row + nt_row, 1);
d_tt = pred(ns_row + 1 : ns_row + nt_row, 2);

disp('Finish Density Estimation....')


end

