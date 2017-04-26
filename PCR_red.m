function [ yhat, a ] = PCR_red( X, y)
% ANI DEVER s225055
% This function computes the Principal Component Regression
% takes 2 matrices, X & y
% computes PCA inside, obtains Z matrix 
% reduces the U and D according to threshold set for eigenvalues
% returns yhat and a 

% Necessary variables 
N = length(X);
R = (X.'*X)/N; % R = E{x*x^T}
[U,D] = eig(R); % diagonal matrix D of eigenvalues and matrix U whose columns are eigenvectors
d = diag(D); % taking the diagonal of matrix
d1 = d/sum(d); % 
d1c = cumsum(d1);
removed_eigen = 1e-4; nrem=(d1c<removed_eigen); % filtering smaller eigenvalues
% reducing dimensionality
UL=U;UL(:,nrem)=[];
DL=D(sum(nrem==1)+1:end,sum(nrem==1)+1:end);
Z_norm = 1/sqrt(N)*X*UL*DL^(-1/2);
% Matrix Z_norm stores a set of F orthonormal column vectors
Zy = Z_norm.'*y; % evaluating the vector Zy
a = 1/N*(UL)*inv(DL)*UL.'*X.'*y;
yhat = Z_norm*Zy;
fprintf('%i eigenvalues(<%f) out of %i are removed\n',sum(nrem==1),removed_eigen,length(R));
end