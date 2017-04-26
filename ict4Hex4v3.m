% % ICT4HEALTH LAB.4 soft k-means
% % ANI DEVER s225055
clear variables; close all; clc,tic;
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
k = 2; % no of clusters
iter = 10;
rng('default'); % fixing the random number generation
%% The soft k-means algorithm
x_k(1,:) = y(201,:);
x_k(2,:) = y(101,:);
var_k = ones(1,k);
pi_k =  1/k*ones(1,k);
% preallocating matrices to use in a loop
responsibility = zeros(N,k);
nom = zeros(N,1);
denom = zeros(N,1);
distance = (pdist2(y,x_k)).^2;
% Assignment step
for ii=1:iter
    for j=1:N
        nom(j,1) = (pi_k(1)*exp(distance(j,1)/(-2*var_k(1))))/(2*pi*var_k(1))^(M/2);
        denom(j,1) = nom(j,1)+((1-pi_k(1))*exp(distance(j,2)/(-2*var_k(2))))/(2*pi*var_k(2))^(M/2);
        %nom(j,2) = (1-pi_k(1)*exp(distance(j,2)/(-2*var_k(2))))/(2*pi*var_k(2))^(M/2);
        %denom(j,2) = nom(j,2)+(pi_k(1)*exp(distance(j,1)/(-2*var_k(1))))/(2*pi*var_k(1))^(M/2);
    end
    responsibility = nom./denom;
    force=find(1e-3>responsibility(:,1)|responsibility>=0.999);
    responsibility(force)=1e-2;
    responsibility(isnan(responsibility)| isinf(responsibility))=1e-2;
    responsibility(:,2) = 1-responsibility(:,1);    
    Rk = (responsibility(:,1)>=responsibility(:,2));
    x_k = zeros(k,M);
    % cluster updates
    for j=1:N
        dummy(1,:) = responsibility(j,1)*y(j,:)/responsibility(j,1);
        x_k(1,:) = x_k(1,:)+dummy(1,:);
        dummy(2,:) = responsibility(j,2)*y(j,:)/responsibility(j,2);
        x_k(2,:) = x_k(2,:)+dummy(2,:);
    end
    clear dummy;
    % variance updates
    for j=1:N
        dummy(1) = (responsibility(j,1)*distance(j,1))/(M*responsibility(j,1));
        var_k(1) = var_k(1)+dummy(1);
        dummy(2) = (responsibility(j,2)*distance(j,2))/(M*responsibility(j,2));
        var_k(2) = var_k(2)+dummy(2);
    end
    % pi updates
    for j=1:N
        if pi_k(1)>0.999 || pi_k(2)>0.999
            continue
        else
            dummy(1) = responsibility(j,1)/responsibility(j,2);
            pi_k(1) = pi_k(1)+dummy(1);
            pi_k(2) = 1-pi_k(1);
        end
         % dummy(2) = responsibility(j,2)/responsibility(j,1);
         % pi_k(2) = pi_k(2)+dummy(2);
    end
    clear dummy;
    distance = (pdist2(y,x_k)).^2;  
end
toc;
%% Comments 
% The last step of Lab4, soft k-means was not succesful. 
% Lots of assumptions should be done, however still the responsibility is
% a breaking point. We are avoiding the values<1e-3, NaNs, Infs,
% and values very close to zero rounded by MATLAB itself
% (I also wanted to avoid the exact probability values (0,1)). After the second
% iteration almost all values needs to forced, filtered. 
% In conclusion I have to say that my soft kmeans algorithm was failed in
% this case. 