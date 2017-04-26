function [ z ] = PCA( X )
% ANI DEVER s225055
% This function computes the Principal Component Analysis 
% takes a matrix as an input(X)
% returns the Z matrix which is Z = X*U
% U and the matrix of eigenvectors of R(the covariance matrix of input matrix)
% and X as input matrix(var1) 
% returns the Z matrix

N = length(X);
R = (X.'*X)/N; % We diagonalize, R = E{x*x^T}
[U,~] = eig(R); % diagonal matrix D(here ~) of eigenvalues and matrixU whose columns are eigenvectors
z = X*U;
end

