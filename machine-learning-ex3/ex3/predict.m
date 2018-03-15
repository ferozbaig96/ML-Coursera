function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

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

% activation values of each layer (using forward propagation)

a1 = X;
% adding bias unit
% now a1 is m by 401
a1 = [ones(size(a1, 1), 1), a1];

% a1 is m by 401
% Theta1 is 25 by 401
% Theta1' is 401 by 25
% now a2 is m by 25
z2 = a1 * Theta1';
a2 = sigmoid(z2);

% adding bias unit
% now a2 is m by 26
a2 = [ones(size(a2, 1), 1), a2];

% a2 is m by 26
% Theta2 is num_labels by 26
% Theta2' is 26 by num_labels
% now a3 is m by num_labels
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% all_prob is of order m by K
% prob = h_thetaX = sigmoid(X * all_theta) in Logistic Regression
all_prob = a3;

% taking max along each row of all_prob
% and returning the [max, label]
% where label is the index of the max found
[~, p] = max(all_prob, [], 2);

% =========================================================================


end
