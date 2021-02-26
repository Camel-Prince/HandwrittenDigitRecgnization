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

%%% forwardpropagation and compute the Cost

X = [ones(m,1),X];
z2 = X*Theta1';
a2 = sigmoid(z2);

a2 = [ones(m,1),a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

H = a3;  

%[max_num, col_num] = max(H,[],2);
%p = col_num;

J = 0;
for i = 1:m
  h = H(i,:);
  tempY = zeros(1,num_labels);
  tempY(y(i)) = 1;
  
  J = J+ sum( tempY.*log(h) + (1.-tempY).*log(1.-h) ) ;
  
  
endfor
%% fprintf('Now the size of J is %f lines %f rows', size(J,1), size(J,2) );

J = -1/m* J;

%% regularize the J:
for i = 1:hidden_layer_size 
  for j = 1: input_layer_size
    J = J + lambda/(2*m)* Theta1(i,j+1)^2;
  endfor
endfor


for i = 1:num_labels
  for j = 1:hidden_layer_size
    J = J + lambda/(2*m) * Theta2(i,j+1)^2;
  endfor
endfor



%% BP and compute grad

big_delta1  = zeros( hidden_layer_size, input_layer_size+1 );  
big_delta2  = zeros( num_labels, hidden_layer_size+1 );

for it = 1:m % For each example:
  a_1 = X(it,:)';
  a_2 = a2(it,:)';
  a_3 = a3(it,:)'; 
  
%fprintf('a_3 size : %f %f',size(a_3,1), size(a_3,2) );

  z_2 = z2(it,:)';
  z_3 = z3(it,:)';
  
  small_delta_3 = a_3;
  small_delta_3( y(it) ) = small_delta_3( y(it) ) - 1;
  
  small_delta_2 = Theta2'([2:hidden_layer_size+1],:) * small_delta_3 .* sigmoidGradient(z_2);
  
  %compute big_delta1 whose size is 25* 401;
  big_delta1 = big_delta1 + small_delta_2 * a_1'; 
  
  %compuute big_delta2 whose size is 10 * 26;
  
  big_delta2 = big_delta2 + small_delta_3 * a_2';
  
endfor

%% unregularized grad:

Theta1_grad = 1/m* big_delta1;
Theta2_grad = 1/m* big_delta2;

%% regularize: 
re_Theta1 = Theta1;
re_Theta1(:,1) = zeros( hidden_layer_size ,1);
Theta1_grad = Theta1_grad + lambda/m* re_Theta1;

re_Theta2 = Theta2;
re_Theta2(:, 1) = zeros(num_labels ,1);
Theta2_grad = Theta2_grad + lambda/m* re_Theta2;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
