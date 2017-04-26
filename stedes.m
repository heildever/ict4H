function [ yhat,a_hat ] = stedes( var1, var2 )
% ANI DEVER s225055
% This function computes the iterative steepest descent algorithm
% the function takes 2 matrices(var1, var2),
% initializes a random vector as initial estimate of â(i),
% initializes a vector of ones to update at each iteration,
% and returns the estimate of y(yhat) and the estimate values a_hat

rng('default'); % fixing the random number generation
[~,N]=size(var1);
a_hat = rand(N,1); % initial vector of â(i)
a_hat_up = rand(N,1); % a dummy vector to be updated at each iteration

while norm(a_hat-a_hat_up)>1e-6 % stop condition 
    gradient_sd = (-2*transpose(var1)*var2)+2*transpose(var1)*var1*a_hat_up;
    Hessian = 4*(var1.'*var1);
    a_hat_up = a_hat;
    a_hat = a_hat_up-(norm(gradient_sd)^2/(transpose(gradient_sd)*Hessian*gradient_sd))*gradient_sd;
end
yhat = var1*a_hat; % estimate of y
end