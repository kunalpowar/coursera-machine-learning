function [J grad] = nnCostFunction(nn_params, ...
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
X = [ones(m, 1) X];

z2 = Theta1*X';
a2 = sigmoid(z2);
a2 = a2';
a2 = [ones(size(a2,1), 1) a2];

z3 = Theta2*a2';
a3 = sigmoid(z3);
% a3 is of size 10 X 5000.

% y is a vector 5000 X 1.
% Need to convert it to binary representation resulting in 5000 X 10.
y_vectors = zeros(m, num_labels);
for iter = 1:size(y, 1)
    y_vectors(iter, y(iter)) = 1;
end
y_vectors = y_vectors';
% y_vectors is size 10 X 5000.
J = sum(((-y_vectors.*log(a3)) - ((1-y_vectors).*log(1-a3)))(:))/m;


% Ignore the bias unit.
temp1 = Theta1;
temp1(:, 1) = 0;
temp2 = Theta2;
temp2(:, 1) = 0;

reg_param = (sum(sum(temp1.^2))) + (sum(sum(temp2.^2)));
J = J + ((lambda/(2*m))*reg_param);

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for iter = 1:m
    a1 = X(iter,:)';
    % a3 = y_vectors(:,iter);
    
    % For layer 2.
    z2 = Theta1*a1; % size(z2) = 25 X 1 (size(Theta1) = 25 X 401 ) 
    a2 = sigmoid(z2); % size(a2) = 25 X 1

    % For layer 3.
    a2 = [1; a2]; % size(a2) = 26 X 1
    z3 = Theta2*a2; % size(z3) = 10 X 1 (size(Theta2) = 10 X 26 ) 
    a3 = sigmoid(z3); % size(a3) = 10 X 1
    
    ym = y_vectors(:,iter); % size(ym) = 10 X 1

    del3 = a3 - ym; % 10 X 1
    % find del2 with generic form del(l) = theta(l)'*del(l+1).*a(l).*(1-a(l))
    % sizes in this case: ((26 X 10)*(10 X 1)).*(26 X 1).*(26X1) = 26 X 1
    del2 = (Theta2'*del3).*(a2).*(1-a2); % 26 X 1

    % generic implementation: delta(l) := delta(l) + del(l+1)*(a(l))'
    delta1 = delta1 + del2(2:end,:)*a1'; % (25 X 1) * (1 X 401) = 25 * 401
    delta2 = delta2 + del3*a2'; % (10 X 1) * (1 X 26) = 10 * 26
end

% Unregularised.
Theta1_grad = (1/m)*(delta1); % 25 * 401
Theta2_grad = (1/m)*(delta2); % 10 * 26 

% Regularised.
% Theta1_grad = [Theta1_grad(:,1), (Theta1_grad(:,2:end) + lambda.*Theta1(:,2:end))/m];
% Theta2_grad = [Theta2_grad(:,1), (Theta2_grad(:,2:end) + lambda.*Theta2(:,2:end))/m];

reg1 = [(lambda/m)*Theta1(:,2:end)];
reg2 = [(lambda/m)*Theta2(:,2:end)];

Theta1_grad = [Theta1_grad(:,1) Theta1_grad(:,2:end).+reg1];
Theta2_grad = [Theta2_grad(:,1) Theta2_grad(:,2:end).+reg2];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
