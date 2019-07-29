function [J, grad] = linearRegCostFunction(X, y, theta, lambda)



m = length(y); % number of training examples


J = 0; % cost function (scalar)
grad = zeros(size(theta)); % gradient vector



h = X * theta;

J = (1 / (2*m)) * sum((h - y).^2) + (lambda / (2*m)) * sum(theta(2:end,:).^2);

theta2 = theta;

theta2(1) = 0;

grad = (1/m) * (X' * (h - y)) + (lambda / m) * theta2;



grad = grad(:);

end
