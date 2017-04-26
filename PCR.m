function [ yhat, a ] = PCR( X, y, Z )
% ANI DEVER s225055
% This function computes the Principal Component Regression
% takes 2 matrices, X & y
% and the Z matrix as input, obtained by performing PCA
% returns yhat and a

% Necessary variables
N = length(X);
R = 1/N*(X.'*X); % R = E{x*x^T}
[~, D] = eig(R); % diagonal matrix D of eigenvalues and matrix V whose columns are eigenvectors
Z_norm = 1/sqrt(N)*Z*D^(-1/2);
% Matrix Z_norm stores a set of F orthonormal column vectors
Zy = Z_norm.'*y; % evaluating the vector Zy
a = (X.'*X)^(-1)*X.'*y;
yhat = Z_norm*Zy;
end