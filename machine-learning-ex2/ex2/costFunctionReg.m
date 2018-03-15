function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% h_thetaX = prob
prob = sigmoid(X*theta);

% For regularization, theta0 is not regularised ( j starts from 1 to n, not
% 0 to n in the equation )
temp_theta1 = theta(1);
theta(1) = 0;

J = (-1 / m) * (y' * log(prob) ...
                + (1 - y)' * log(1 - prob)) ...
                ...regularization
                + (lambda / (2 * m)) * (theta' * theta);

grad = (1 / m) * X' * (prob - y) + (lambda / m) * theta;

% OR
% J = (-1 / m) * sum( log(prob) .* y ...
%                   + log(1 - prob) .* (1 - y) ) ...
%                   ...regularization
%                   + (lambda / (2 * m)) * sum (theta .^ 2);
  
% OR
% for j = 1 : size(theta, 1)  % rows (n+1)
%     grad(j) = (1 / m) * sum((prob - y) .*  X(: , j)) + (lambda / m) * theta(j);
% end

theta(1) = temp_theta1;


% =============================================================

end
