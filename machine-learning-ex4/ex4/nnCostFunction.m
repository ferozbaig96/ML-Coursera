function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% FORWARD PROPAGATION
% activation values of each layer (using feedforward propagation)

a1 = X;
% adding bias unit
a1 = [ones(size(a1, 1), 1), a1];
% a1 is m by (input_layer_size + 1)
% Theta1 is hidden_layer_size by (input_layer_size + 1)
% Theta1' is (input_layer_size + 1) by hidden_layer_size
% now a2 is m by hidden_layer_size
z2 = a1 * Theta1';
a2 = sigmoid(z2);

% adding bias unit
a2 = [ones(size(a2, 1), 1), a2];
% a2 is m by (hidden_layer_size + 1)
% Theta2 is num_labels by (hidden_layer_size + 1)
% Theta2' is (hidden_layer_size + 1) by num_labels
% now a3 is m by num_labels
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% computing y as per multi-class classification output (0 or 1 only)
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

% OR
% y_matrix = zeros(size(a3));
% for i = 1 : size(a3,1)
%     y_matrix(i, y(i)) = 1;
% end

% For regularization, bias unit is not regularized ( j starts from 1 to n, not
% 0 to n in the equation )
temp_theta11 = Theta1(:, 1);
Theta1(:, 1) = 0;
temp_theta21 = Theta2(:, 1);
Theta2(:, 1) = 0;

h_ThetaX = a3;

term1 = y_matrix .* log(h_ThetaX) + (1 - y_matrix) .* log(1 - h_ThetaX);
% can't use transpose as Theta is a matrix (not a vector)
term2 = Theta1 .^ 2;
term3 = Theta2 .^ 2;

J = (-1 / m) * sum(term1(:)) ...
    ... regularization
    + (lambda / (2 * m)) * (sum(term2(:)) + sum(term3(:)));

Theta1(:, 1) = temp_theta11;
Theta2(:, 1) = temp_theta21;

% BACKWARD PROPAGATION

Theta1_grad = 0;
Theta2_grad = 0;

for t = 1 : m
    % computing activations
    a1 = [1, X(t, :)]';     % 401 by 1
    z2 = Theta1 * a1;       % 25 by 1
    a2 = [1; sigmoid(z2)];  % 26 by 1
    z3 = Theta2 * a2;       % 10 by 1
    a3 = sigmoid(z3);       % by by 1
   
    % computing errors
    delta_3 = a3 - y_matrix(t, :)';     % 10 by 1
    delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];  % 26 by 1
    % delta_1 is not calculated because we do not associate error with the input    
    % Taking off the bias row
    delta_2 = delta_2(2:end);           % 25 by 1
        
    % accumulating gradient
    Theta1_grad = Theta1_grad + delta_2 * a1';  % 25 by 401
    Theta2_grad = Theta2_grad + delta_3 * a2';  % 10 by 26
    
end

% For regularization, bias unit is not regularized ( j starts from 1 to n, not
% 0 to n in the equation )
temp_theta11 = Theta1(:, 1);
Theta1(:, 1) = 0;
temp_theta21 = Theta2(:, 1);
Theta2(:, 1) = 0;

% computing gradients
Theta1_grad = (1 / m) * (Theta1_grad + lambda * Theta1);
Theta2_grad = (1 / m) * (Theta2_grad + lambda * Theta2);

Theta1(:, 1) = temp_theta11;
Theta2(:, 1) = temp_theta21;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
