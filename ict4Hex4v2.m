% % ICT4HEALTH LAB.4 hard k-means
% % ANI DEVER s225055
clear variables; close all; clc,tic;
%% preparing the data 
load('arrhythmia.mat');
arrhythmia = arrhythmia(:,any(arrhythmia)); % removing the columns with only 0's
class_id = arrhythmia(:,end); % vector of classes in given order
iii = find(class_id(:,end)>1); % find classes >1
class_id(iii,end)=2; % equalize them to 2         
y = arrhythmia(1:end,1:(end-1)); % y is a matrix without the class data
y = normalize(y); % normalizing y matrix
% Defining the variables
[N, M] = size(y);
iter = 9; % no of iterations
rng('default'); % fixing the random number generation
%% The hard K-means algorithm with 4 pseudo-random initial vectors
k = 4; % number of clusters
% x_k = randn(k,M); % initial random vector of size Mxk
x_k(1,:) = y(51,:);
x_k(2,:) = y(101,:);
x_k(3,:) = y(151,:);
x_k(4,:) = y(201,:);
var_k = ones(1,k);
pi_k =  1/k*ones(1,k);
distance = zeros(1,k); % preallocating distance vector
for jj = 1:iter
    dec=zeros(N,1);
    for i=1:N
        distance(1)=(norm(y(i,:)-x_k(1,:)))^2;
        distance(2)=(norm(y(i,:)-x_k(2,:)))^2;
        distance(3)=(norm(y(i,:)-x_k(3,:)))^2;
        distance(4)=(norm(y(i,:)-x_k(4,:)))^2;
        [~,I] = min(distance);% index of the centroid at min distance is I
        dec(i)=I;% pixel B(i,:) is given to cluster I
    end
    Wnk = y(dec==1,:);
    Wnj = y(dec==2,:);
    Wnh = y(dec==3,:);
    Wng = y(dec==4,:);
    pi_k(1) = size(Wnk,1)/N;
    pi_k(2) = size(Wnj,1)/N;
    pi_k(3) = size(Wnh,1)/N;
    pi_k(4) = size(Wng,1)/N;
    x_k(1,:) = sum(Wnk,1)/size(Wnk,1);
    x_k(2,:) = sum(Wnj,1)/size(Wnj,1);
    x_k(3,:) = sum(Wnh,1)/size(Wnh,1);
    x_k(4,:) = sum(Wng,1)/size(Wng,1);
    for j=1:size(Wnk,1)
        dummy = (norm(Wnk(j,:)-x_k(1,:)))^2;
        var_k(1) = var_k(1)+dummy;
    end
    var_k(1) = var_k(1)/((size(Wnk,1)-1)*M);
    for j=1:size(Wnj,1)
        dummy = (norm(Wnj(j,:)-x_k(2,:)))^2;
        var_k(2) = var_k(2)+dummy;
    end
    var_k(2) = var_k(1)/((size(Wnj,1)-1)*M);
    for j=1:size(Wnh,1)
        dummy = (norm(Wnh(j,:)-x_k(3,:)))^2;
        var_k(3) = var_k(3)+dummy;
    end
    var_k(3) = var_k(3)/((size(Wnh,1)-1)*M);
    for j=1:size(Wng,1)
        dummy = (norm(Wng(j,:)-x_k(4,:)))^2;
        var_k(4) = var_k(4)+dummy;
    end
    var_k(4) = var_k(4)/((size(Wng,1)-1)*M);    
end
fprintf('According doctors diagnosis  =>\n');
fprintf('Total no of healthy patients: %i\n',sum(class_id==1));
fprintf('Total no of arrhythmic patients %i\n',sum(class_id==2));
fprintf('According to coded hard k-means algorithm  =>\n');
fprintf('Total no of cluster -1- patients: %i\n',sum(dec==1));
fprintf('Total no of cluster -2- patients %i\n',sum(dec==2));
fprintf('Total no of cluster -3- patients %i\n',sum(dec==3));
fprintf('Total no of cluster -4- patients %i\n',sum(dec==4));
error_1 = immse(dec,class_id);
fprintf('The msee of 4 cluster hard kmeans: %f\n',error_1);
figure(),plot(sortrows(class_id),'bo'), grid on, hold on,
plot(sortrows(dec),'rx'),title('cluster distribution');
%% MATLAB embedded k-means algorithm
idx = kmeans(y,k);
fprintf('According to MATLAB emdedded hard k-means algorithm  =>\n');
fprintf('Total no of cluster -1- patients: %i\n',sum(idx==1));
fprintf('Total no of cluster -2- patients %i\n',sum(idx==2));
fprintf('Total no of cluster -3- patients %i\n',sum(idx==3));
fprintf('Total no of cluster -4- patients %i\n',sum(idx==4));
error_2 = immse(idx,class_id);
fprintf('The msee of 4 cluster MATLAB hard kmeans: %f\n',error_2);
figure(),plot(sortrows(class_id),'bo'), grid on, hold on,
plot(sortrows(idx),'rx'),title('cluster distribution k-means MATLAB');
toc; 
%% Comments
% This script computes the 4 class case of the hard k-means algorithm.
% The intermediate steps are the same as 2 class case and the algorithm
% gets saturated at 9th iteration. The hard k-means algorithm provided by
% MATLAB is added in order to provide comparison.
% This time my script provides relatively better results than MATLAB
% kmeans. 