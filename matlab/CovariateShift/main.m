%%Even though the method is called Binary/MultiClassRobustTrain/Test, the
%%baseline methods: LR and IW, can also be implemented via the robust
%%framework. The LR method is just plugging in all ones for both r_ts and
%%r_st. IW method is setting the r_st to ones and set r_ts to true paras,
%%while Robust method is setting r_ts to ones.
%%See the below for examples. 
%%In Binary 
%%cases, sample data are sampled from a multivariate gaussian, mu and 
%%var are known beforehand.

%load 2-dim gaussian data
load('data/gaussian1/y_1.mat');
load('data/gaussian1/x_1.mat');
load('data/gaussian1/y_2.mat');
load('data/gaussian1/x_2.mat');

y_1 = y_1';
y_2 = y_2';

%get density under gaussian
mu_s = [6, 6];
var_s = [3, -2; -2, 3];
mu_t = [7, 7];
var_t = [3, 2; 2, 3];

%Robust
d_s = mvnpdf(x_1, mu_s, var_s);
d_t = mvnpdf(x_1, mu_t, var_t);
theta_1 = BinaryRobustTrain(x_1, y_1, d_s./d_t, ones(size(x_1,1),1));
%LR
theta_2 = BinaryRobustTrain(x_1, y_1, ones(size(x_1,1),1), ones(size(x_1,1),1));
%IW
r_ts = d_t./d_s;
theta_3 = BinaryRobustTrain(x_1, y_1, ones(size(x_1,1),1), r_ts);

%Testing
%Robust
d_s = mvnpdf(x_2, mu_s, var_s);
d_t = mvnpdf(x_2, mu_t, var_t);
[logloss_1, pred_1] = BinaryRobustTest(theta_1, x_2, y_2, d_s./d_t);
%LR
[logloss_2, pred_2] = BinaryRobustTest(theta_2, x_2, y_2, ones(size(x_2,1),1));
%IW
[logloss_3, pred_3] = BinaryRobustTest(theta_3, x_2, y_2, ones(size(x_2,1),1));
%Accuracy
acc_1 = ComputeAcc(pred_1, y_2);
acc_2 = ComputeAcc(pred_2, y_2);
acc_3 = ComputeAcc(pred_3, y_2);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%multiclass classification

load('data/iris/iris_train.mat');
load('data/iris/iris_test.mat');

X_s = iris_train(:, 1:end-1);
X_t = iris_test(:, 1:end-1);
y_s = iris_train(:, end);
y_t = iris_test(:, end);

[d_ss, d_st, d_ts, d_tt] = LRDensityEstimation(X_s, X_t, 'lambdas', [ 0.1 1 10 ]);

[ns_row, n_col] = size(X_s);
[nt_row, ~] = size(X_t);
n_class = max(y_s);

%construct lambda vector
lambda = 2 * std(X_s)/sqrt(ns_row);
lambda(1) = 1;

theta = MultiClassRobustTrain(X_s, y_s, n_class, d_ss./d_st,  ones(ns_row, 1), 'lambda', lambda, 'rate', 1, 'min_gradient', 0.01 );
[logloss, pred] = MultiClassRobustTest(theta, X_t, y_t, n_class, d_ts./d_tt)
acc = ComputeAcc(pred, y_t);
msg = sprintf('Acc is %f and logloss is %f for robust method', acc, logloss); disp(msg);

theta = MultiClassRobustTrain(X_s, y_s, n_class, ones(ns_row, 1), ones(ns_row, 1), 'lambda', lambda, 'rate', 1, 'min_gradient', 0.01 );
[logloss, pred] = MultiClassRobustTest(theta, X_t, y_t, n_class, ones(nt_row, 1))
acc = ComputeAcc(pred, y_t);
msg = sprintf('Acc is %f and logloss is %f for IR method', acc, logloss); disp(msg);

r_ts = d_st./d_ss;
theta = MultiClassRobustTrain(X_s, y_s, n_class, ones(ns_row, 1), r_ts, 'lambda', lambda, 'rate', 1, 'min_gradient', 0.01 );
[logloss, pred] = MultiClassRobustTest(theta, X_t, y_t, n_class, ones(nt_row, 1))
acc = ComputeAcc(pred, y_t);
msg = sprintf('Acc is %f and logloss is %f for IW method', acc, logloss); disp(msg);




