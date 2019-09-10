function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values

m = length(y); % number of training examples

% You need to return the following variables correctly 
#inside this fuction we have to use sigmoid function

J = 0;
grad = zeros(size(theta)); #n x 1 matrix where n= #of features
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%





#inside this fuction we have to use sigmoid function
# +++++++++++++COPY THE week 2 regularized costFunction() and paste here++++++++++++++++
% Initialize some useful values


H_theta_X = sigmoid(X*theta); # m x 1 matrix where m = # of training set

# Regularization term
reg = zeros(size(theta)-1); # excluding bais theta_0
theta_biased_removed = theta;
theta_biased_removed(1,:) = []; #removing theta_0 bised term
reg= (lambda / (2*m))*sum((theta_biased_removed .^2)); 

cost_of_H_theta_X_comma_Y = - (y .* log(H_theta_X) + (1-y) .* log(1- H_theta_X)); 
# here cost_of_H_theta_X_comma_Y  is cost(H_theta(X), Y) where H_theta_X is m x 1 and Y is m x 1 matrix
#cost_of_H_theta_X_comma_Y  is m x 1 matrix
J= sum(cost_of_H_theta_X_comma_Y)/m + reg; 

# finding gradient called delta
delta= zeros(length(theta),1); % (n+1)x 1 matrix
    for i= 1: length(theta),
      if i==1,
        delta(i) = (H_theta_X - y)' * X(:,i);
      else
        delta(i) = (H_theta_X - y)' * X(:,i) + lambda * theta(i); # delta (n X 1); delta/m
      endif
        
    endfor
grad= delta/m; # as m is found when we take partial derivatives





% =============================================================


end
