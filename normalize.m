function [ output_matrix ] = normalize( input_matrix )
% ANI DEVER s225055
% This function normalizes the input_matrix
% gives a normalized matrix as output_matrix
[M, ~] = size(input_matrix);
o = ones(M,1);
output_matrix = (input_matrix-o*mean(input_matrix))./sqrt(o*var(input_matrix));
end

