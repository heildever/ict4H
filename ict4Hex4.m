% % ICT4HEALTH LAB.4
% % ANI DEVER s225055
clear variables, close all; clc, tic;
%% preparing the data 
load('arrhythmia.mat');
arrhythmia = arrhythmia(:,any(arrhythmia)); % removing the columns with only 0's
class_id = arrhythmia(:,end); % vector of classes in given order
classhat = zeros(size(class_id)); % preallocating class estimate vector
iii = find(class_id(:,end)>1); % find classes >1
class_id(iii,end)=2; % equalize them to 2         
y = arrhythmia(1:end,1:(end-1)); % y is a matrix without the class data
y = normalize(y); % normalizing y matrix
% Defining the variables
[N, M] = size(y);
iter = 15; % iteration counter
k = 2; % number of clusters
fprintf('According doctors diagnosis  =>\n');
fprintf('Total no of healthy patients: %i\n',sum(class_id==1));
fprintf('Total no of arrhythmic patients %i\n',sum(class_id==2));
rng('default'); % fixing the random number generation
% initializing cluster centroids
%% The hard K-means algorithm with the vector from Lab3
load('dist2');
distance = dist2;
var_k = ones(1,k);
pi_k =  1/k*ones(1,k);
for i=1:iter
    dec=zeros(N,1);    
    for j=1:N
        [~,I] = min(distance(j,:));% index of the centroid at min distance is I
        dec(j)=I;% pixel B(i,:) is given to cluster I
    end
    Wnk = y(dec==1,:);
    Wnj = y(dec==2,:);
    pi_k(1) = size(Wnk,1)/N;
    pi_k(2) = size(Wnj,1)/N;
    x_k(1,:) = sum(Wnk,1)/size(Wnk,1);
    x_k(2,:) = sum(Wnj,1)/size(Wnj,1);
    for j=1:size(Wnk,1)
        dummy = (norm(Wnk(j,:)-x_k(1,:)))^2;
        var_k(1) = var_k(1)+dummy;
    end
    var_k(1) = var_k(1)/((size(Wnk,1)-1)*M);
    for j = 1:size(Wnj,1)
        dummy = (norm(Wnj(j,:)-x_k(2,:)))^2;
        var_k(2) = var_k(2)+dummy;
    end
    var_k(2) = var_k(2)/((size(Wnj,1)-1)*M);
    for j=1:N
        distance(j,1)=(norm(y(j,:)-x_k(1,:)))^2;
        distance(j,2)=(norm(y(j,:)-x_k(2,:)))^2; 
    end
end
fprintf('According to hard k-means algorithm with Xk vector from lab ex3 =>\n');
fprintf('Total no of clustered healthy patients: %i\n',sum(dec==1));
fprintf('Total no of clustered arrhythmic patients: %i\n',sum(dec==2));
error_1 = immse(dec,class_id);
fprintf('The msee of 2 cluster hard kmeans(vector from Lab3): %f\n',error_1);
figure(),plot(sortrows(class_id),'bo'), grid on, hold on,
title('clustering using the vector from Lab3'),plot(sortrows(dec),'rx');
%% The hard K-means algorithm with 2 pseudo-random initial vectors 
% x_k = randn(k,M); % initial random vector of size Mxk
x_k(1,:) = y(50,:);
x_k(2,:) = y(150,:);
var_k = ones(1,k);
pi_k =  1/k*ones(1,k);
distance = zeros(1,k); % preallocating distance vector
for ij = 1:iter
    dec=zeros(N,1);
    for i=1:N
        distance(1)=(norm(y(i,:)-x_k(1,:)))^2;
        distance(2)=(norm(y(i,:)-x_k(2,:)))^2;
        [~,I] = min(distance);% index of the centroid at min distance is I
        dec(i)=I;   % vector B(i,:) is given to cluster I
    end
    Wnk = y(dec==1,:);
    Wnj = y(dec==2,:);
    pi_k(1) = size(Wnk,1)/N;
    pi_k(2) = size(Wnj,1)/N;
    x_k(1,:) = sum(Wnk)/size(Wnk,1);
    x_k(2,:) = sum(Wnj)/size(Wnj,1);
        for j = 1:size(Wnk,1)
            dummy = norm(Wnk(j,:)-x_k(1,:)).^2;
            var_k(1) = var_k(1)+dummy;
        end
        var_k(1) = var_k(1)/((size(Wnk,1)-1)*M);
        for j = 1:size(Wnj,1)
            dummy = norm(Wnj(j,:)-x_k(2,:)).^2;
            var_k(2) = var_k(2)+dummy;
        end
        var_k(2) = var_k(2)/((size(Wnj,1)-1)*M);
end
fprintf('According to hard k-means algorithm with 2 pseudo-random initial vectors =>\n');
fprintf('Total no of clustered healthy patients: %i\n',sum(dec==1));
fprintf('Total no of clustered arrhythmic patients: %i\n',sum(dec==2));
error_2 = immse(dec,class_id);
fprintf('The msee of 2 cluster hard kmeans: %f\n',error_2);
figure(),plot(sortrows(class_id),'bo'), grid on, hold on,
title('clustering using pseudo-random vectors'),plot(sortrows(dec),'rx');
%% MATLAB emdedded k-means algorithm
idx = kmeans(y,k);
fprintf('According to MATLAB emdedded hard k-means algorithm  =>\n');
fprintf('Total no of clustered healthy patients: %i\n',sum(idx==1));
fprintf('Total no of clustered arrhythmic patients: %i\n',sum(idx==2));
error_3 = immse(idx,class_id);
fprintf('The msee of MATLAB hard kmeans: %f\n',error_3);
figure(),plot(sortrows(class_id),'bo'), grid on, hold on,
plot(sortrows(idx),'rx'),title('cluster distribution k-means MATLAB');
toc; 
%% Comments 
% This exercise uses the same dataset(arrhythmia) as used in Lab3, however
% the difference is this exercise studies clustering instead of
% classification. 
% There different clustering approaches were observed among the script. The first one,
% using the vector from Lab ex3, second one with using pseudo-random
% vectors. The algorithm steps(assignment, update of variances and probabilities then clusters) are followed as
% described in the related course material. The minimum iteration number is found as 15(variable 'iter'). 
% The third approach is provided by MATLAB itself, the command k-means
% classifies the given data into specified number of clusters.
% MATLAB explanation provided for embedded k-means algorithm: 
% kmeans by MATLAB uses a two-phase iterative algorithm to minimize the sum of point-to-centroid distances,
% summed over all k clusters. This first phase uses batch updates, where each iteration consists of reassigning
% points to their nearest cluster centroid, all at once, followed by recalculation of cluster centroids.
% This phase occasionally does not converge to solution that is a local minimum.
% That is, a partition of the data where moving any single point to a different cluster increases the total
% sum of distances. This second phase uses online updates, where points are individually reassigned if doing
% so reduces the sum of distances, and cluster centroids are recomputed after each reassignment.
% Each iteration during this phase consists of one pass though all the points.
% This phase converges to a local minimum, although there might be other local minima with lower total sum of distances.
% In general, finding the global minimum is solved by an exhaustive choice of starting points,
% but using several replicates with random starting points typically results in a solution that is a global minimum.
% Personal observations: it was interesting to see even if I have
% considered the vectors directly from data matrix, this case produced the
% poorest results. Then MATLAB given algorithm magically performs best,
% however we have no solid explanation provided by MATLAB. The numbers
% clustered by these 2 algorithms(the one using the vector from Lab3 and MATLAB)
% are very close. 