function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
% Add ones to the X data matrix
X = [ones(m, 1) X]; 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%Z2 = Theta1 X   Theta1 = 25 x 401 and X = 5000 x 401
Z2= Theta1 * X'; # Z2= 25 x 5000 return Z2 for 5000 training sets

A2= sigmoid(Z2); # return 25 x 5000 matrix

A2= [ones(1,m); A2]; # add bias term 1 to to A2 for 5000 training data
% Theta2 has size 10 x 26 

Z3= Theta2 * A2;

A3 = sigmoid(Z3);
[MAX INDICES] = max(A3); # taking maximum of each column
row_p = INDICES; #get the index which is the class label
p= row_p'; 







% =========================================================================


end
