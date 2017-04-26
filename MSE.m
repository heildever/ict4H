function [ yhat, a_hat ] = MSE ( var1, var2 )
% ANI DEVER s225055
% This function computes the minimum square error method 
% the function takes 2 matrices as input
% and returns the estimate of y and the estimate values
a_hat = pinv(var1)*var2;
yhat = var1*a_hat;
% gradient = (-2*transpose(X_train)*y_train)+2*transpose(X_train)*X_train*a_hat_mse;
end

