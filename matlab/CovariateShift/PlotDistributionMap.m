function [ output_args ] = PlotDistributionMap(x_1, y_1, mu_s, var_s, mu_t, var_t)
% (Only for two dimensional multivariate gaussian data and binary classification case)
% Plot scatter, distribution ecllipse and most importantly, the predicition
% distribution colormap, given source datapoints, source gaussian
% distribution paras and target gaussian distribution paras

%   x_1:        training data x, n_row * 2 matrix        
%   y_1:        training labels, 1 or -1 for binary, n_row * 1 vector
%   mu_s:       mu of the source multivariate gaussian distribution
%   var_s:      cov of the source multivariate gaussian distribution
%   mu_t:       mu of the target multivariate gaussian distribution
%   var_t:      cov of the target multivariate gaussian distribution
%   multiview:  0 or 1, whether using multiview method

d_s = mvnpdf(x_1, mu_s, var_s);
d_t = mvnpdf(x_1, mu_t, var_t);


%theta = BinaryRobustTrain(x_1, y_1, d_s./d_t, ones(size(x_1, 1), 1));
%theta = BinaryRobustTrain(x_1, y_1, ones(size(x_1, 1), 1), ones(size(x_1, 1), 1));
r_ts = d_t./d_s;
theta = BinaryRobustTrain(x_1, y_1, ones(size(x_1, 1), 1), r_ts);



%maxs = ceil(max(max(x_1)));
%mins = floor(min(min(x_1)));
maxs = 15;
mins = -5;
% generate "testing" data to plot probability map
[X_dim1, X_dim2] = meshgrid(mins : 0.1 : maxs, mins : 0.1 : maxs);
% get dimension of X_dim1 and X_dim2
dim = (maxs - mins)/0.1 + 1;


prediction = zeros(dim, dim);
%predicting
for i = 1 : dim
    for j = 1 : dim
        x_t = [X_dim1(i, j), X_dim2(i, j)];
        %[~, pred_temp] = BinaryRobustTest(theta, x_t, 1, mvnpdf(x_t, mu_s, var_s)/mvnpdf(x_t, mu_t, var_t));
        [~, pred_temp] = BinaryRobustTest(theta, x_t, 1, 1);     
        prediction(i, j) = pred_temp(1);
    end
end

x_pos = x_1(find(y_1 == 1), :);
x_neg = x_1(find(y_1 == -1), :);



pcolor(X_dim1, X_dim2, prediction);
shading interp
colormap(jet);
colorbar;
set(gca, 'cameraposition', [0 0 180]);  % this essentially turns a 3d surface plot into a 2d plot
cameratoolbar('SetCoordSys', 'y');

hold on

figure = plot(x_pos(:, 1), x_pos(:, 2), 'k+', x_neg(:, 1), x_neg(:, 2), 'ko');
set(figure(1), 'MarkerSize', 8);
set(figure(1), 'LineWidth', 2);

set(figure(2), 'MarkerSize', 8);
set(figure(2), 'MarkerFaceColor', 'w');
hold on

figure = DrawEcllipse(mu_s', 5 * var_s, 'm');
set(figure, 'LineWidth', 2);
figure = DrawEcllipse(mu_t', 5 * var_t, 'm');
set(figure, 'LineWidth', 2);
set(figure, 'LineStyle', '--');

axis off
box off


end

