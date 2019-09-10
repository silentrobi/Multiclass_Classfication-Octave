function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); # p stands for probablity

% Add ones to the X data matrix
X = [ones(m, 1) X]; #m x (n+1) matrix

#Now each training data is passed to all_theta matrix and check which one gives maximum probability. then take that  row number which corresponds to class name. and set to p(m X1 matrix).

TEMP = all_theta * X';   #all theta = k x(n+1) and X' = (n+1)x m. So, temp = k x m

[MAX INDICES] = max(TEMP);
row_p = INDICES; # we need the index which represent the class label
#No need to replace in this dataset as 0 is already replaced to 10 in the dataset
# now replace 10'th index value with 0. Note: index corresponds to class.
#a= [1 2 3 4 1 1 2]
#a(1,(a(1,:)== 1)) = 5
#a =    5   2   3   4   5   5   2

p= row_p'; # contains predicted class 







% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       







% =========================================================================


end
