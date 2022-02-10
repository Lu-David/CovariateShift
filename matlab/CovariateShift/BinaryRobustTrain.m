function [theta] = BinaryRobustTrain(X_s, y_s, r_st, r_ts, varargin)
%Training algorithm for robust binary classification under covariate
%shift(using first order features)
%INPUT:
%       X_s:        training data, a n_row * n_col dimension matrix(without labels)
%       y_s:        training labels, a n_row * 1 vector, in this binary
%                   case, either 1 or -1
%       r_st:        each training data's probability(density) under source distribution, over 
%                    each training data's probability(density) under target distribution, a n_row * 1 vector
%       r_ts:      a para for IW method, which is the inverse of r_st,
%                   set to all ones for other methods
%       varargin:       'lambda'    => the regularization constant
%                       'rate'   => the learning rate
%                       'max_itr' => the max number of iteration
%                       'min_gradient'=> the min number of norm-2 of
%                        gradient(the algorithm will stop when either of the condition is reached first)
%OUTPUT:
%       theta:          the theta parameter, a (n_col + 1) * 1 vector     


% default options 
lambda = 0.001;  % regularization constant
rate = 1; % learning rate
max_itr = 10000;
min_gradient= 0.0001;


% optional var-argin
for i = 1 : 2 : length(varargin)
    name = varargin{i};
    value = varargin{i+1};
    switch name
        case 'lambda'
            lambda = value;
        case 'rate'
            rate = value;
        case 'max_itr'
            max_itr = value;
        case 'min_gradient'
            min_gradient = value;
        otherwise
    end
end

%%get dim of data 
[n_row, n_col] = size(X_s);

%%costruct features, first order features in this case
F = [ones(n_row, 1) X_s];% features are y,X_1*y,X_2*y,...X_n*y
F_g = F .* repmat(r_ts, 1, n_col + 1);
%%initialization before training
P = zeros(n_row, 1);
S_g = ones(n_col + 1, 1) * 10^(-8);%prevent dividing by zero
t = 1;
theta = ones(n_col + 1, 1);
l_0 = 0;
l_1 = (1+sqrt(1+4*l_0^2))/2;
delta_1 = 0;

%%start training
while 1
    t = t + 1;
    decay = sqrt(1000 / (1000 + t));
    l_2 = (1 + sqrt(1 + 4 * l_1^2)) / 2;
    l_3 = (1 - l_1)/l_2;
    for i = 1 : n_row
        W = r_st(i);
        temp = W * theta' * F(i, :)' * y_s(i);
        temp_max = max(temp, -temp);
        temp_min = min(temp, -temp);
        P(i) = exp(temp - temp_max - log(1 + exp(temp_min - temp_max))); % avoiding overflow or underflow
    end
    G =((P.* y_s)' * F_g)' - (y_s' * F_g)' + 2 * lambda .* theta;
    
    if norm(G) < min_gradient% convergence threshold
        disp('Optimization stops by reaching minimum gradient.')
        break;
    elseif t > max_itr
        disp('Optimization stops by reaching maximum iteration.')
        break;
    end
    S_g = S_g + G.^2; % for adaptive gradient  
    delta_2 = theta - decay * rate .* G./ sqrt(S_g);  % adaptive gradient and Nesterov?s Accelerated Gradient Descent
    theta = (1 - l_3) * delta_2 + l_3 * delta_1;
    delta_1 = delta_2;
    l_1 = l_2;
end



