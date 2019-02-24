function [MSE, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation

%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 

grad = zeros(size(theta));

theta(1) = 0;

h= sigmoid(X * theta);

MSE = -(1/m).*sum(y.*log(h) + (1 - y).*log(1 - h)) + (lambda/(2*m)).*sum(theta(2:end).^2);

% =============================================================

grad(1,1) = (1/m)*(X(:,1)'*(h-y)); 
grad(2:end,1)=((1/m)*(X(:,2:end))'*(h-y))+(lambda/(2*m))*theta(2:end);

grad = grad(:);

end
