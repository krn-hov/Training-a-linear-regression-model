function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve



m = size(X, 1); % Number of training examples


error_train = zeros(m, 1);
error_val   = zeros(m, 1);


for i = 1:m
    
    theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
    [J,~] = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
    [Jval,~] = linearRegCostFunction(Xval, yval, theta, 0);
    error_train(i) = J;
    error_val(i) = Jval;
end



end
