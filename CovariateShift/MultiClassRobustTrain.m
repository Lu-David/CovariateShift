function [ theta ] = MultiClassRobustTrain(X_s, y_s, n_class, r_st, r_ts, varargin)
% Training for multiclass robust classification problem
%INPUT:
%        X_s:      training data, a n_row * n_col dimension matrix(without labels)
%        y_s:      training labels, a n_row * 1 vector, in this multiclass
%                   case, range from 1 to N, continuously
%        n_class:   number of class, a scalar
%        r_st:      each training data's probability(density) under source distribution, over 
%                    each training data's probability(density) under target distribution, a n_row * 1 vector
%        r_ts:      a para for IW method, which is the inverse of r_st,
%                   set to all ones for other methods
%        varargin:      'lambda'    => the regularization constant
%                                      a 1*n_col vector            
%                       'rate'   => the learning rate
%                       'max_itr' => the max number of iteration
%                       'min_gradient'=> the min number of norm-2 of
%                        gradient(the algorithm will stop when either of the condition is reached first)
%OUTPUT: 
%        theta:            the theta parameter, a  n_class *n_col matrix   
%

% default options 
lambda = 0.1;  % regularization constant
rate = 0.01; % learning rate
max_itr = 100000;
min_gradient= 0.001;


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
F = X_s; %first order features
F_g = F .* repmat(r_ts, 1, n_col);
Y = zeros(n_row ,n_class);
for i = 1 : n_row
    Y(i, y_s(i)) = 1;
end
lambda = repmat(lambda, n_class, 1);


%%initialization before training
P = zeros(n_row, n_class);
S_g = ones(n_class, n_col) * 10^(-8);%prevent dividing by zero
t = 1;
theta = ones(n_class, n_col);
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
        exp_temp = theta * F(i, :)' * r_st(i);
 
        exp_temp = exp_temp - max(exp_temp); %avoiding overflow or underflow
       
        sum_exp = sum(exp(exp_temp));
        P(i, :) = exp(exp_temp - log(sum_exp));
    end
    
    G = (P' - Y') * F_g + 2 * lambda .* theta;
    
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

