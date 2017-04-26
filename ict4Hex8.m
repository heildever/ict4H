% % ICT4HEALTH LAB.8
% % ANI DEVER s225055
clear variables; close all; clc,tic;
A = imread('Lab8_Health/images/melanoma_12.jpg');
[N1, N2, N3] = size(A);
N = N1*N2;  % N is the total number of pixels
B = double(reshape(A,N,N3));
kcluster = 4; 
[N, M] = size(B);
% B = normalize(B);
Bnew = zeros(N, M);
Bnew2 = zeros(N, M);
% initializing cluster centroids
rng('default');
% x_k = randn(k,M);
x_k(1,:) = B(50,:);
x_k(2,:) = B(550,:);
x_k(3,:) = B(1050,:);
x_k(4,:) = B(1550,:);
distance = zeros(1,kcluster); % preallocating the vector
Niter = 5; % iteration counter
for nit=1:Niter
    dec=zeros(N,1);
    for i=1:N
        distance(1)=(norm(B(i,:)-x_k(1,:)))^2;%square distance w.r.t. centroid #1
        distance(2)=(norm(B(i,:)-x_k(2,:)))^2;%square distance w.r.t. centroid #2
        distance(3)=(norm(B(i,:)-x_k(3,:)))^2;%square distance w.r.t. centroid #3
        distance(4)=(norm(B(i,:)-x_k(4,:)))^2;%square distance w.r.t. centroid #4
        [Y,I] = min(distance);% index of the centroid at min distance is I
        dec(i)=I;% pixel B(i,:) is given to cluster I
    end
    xnew=zeros(kcluster,3);% update the centroids, using the current clusters
    for kk=1:kcluster
        indexes=find(dec==kk);
        xnew(kk,:)=floor(mean(B(indexes,:)));
        Bnew(indexes,:)=ones(length(indexes),1)*xnew(kk,:);
    end
    x_k=xnew;
%     Bnew=floor(Bnew);
%     Anew=reshape(uint8(Bnew),N1,N2,N3);
%     figure(),
%     subplot(1,2,1),imshow(A),title('original image'),
%     subplot(1,2,2),imshow(Anew),title('clustered image');
end
idx = kmeans(B,kcluster); % kmeans by MATLAB
x_k(1,:) = B(50,:);
x_k(2,:) = B(550,:);
x_k(3,:) = B(1050,:);
x_k(4,:) = B(1550,:);
for i=1:N
    if idx == 1
        Bnew2(i,:) = x_k(1,:);
    elseif idx == 2 
        Bnew2(i,:) = x_k(2,:);
    elseif idx == 3 
        Bnew2(i,:) = x_k(3,:);
    elseif idx == 4 
        Bnew2(i,:) = x_k(4,:);        
    end        
end
Bnew=floor(Bnew);
Bnew2=floor(Bnew2);
Anew=reshape(uint8(Bnew),N1,N2,N3);
Anew2=reshape(uint8(Bnew),N1,N2,N3);
figure(),subplot(1,3,1),imshow(A),title('original image'),
subplot(1,3,2),imshow(Anew),title('clustered image'),
subplot(1,3,3),imshow(Anew2),title('kmeans by MATLAB');
%C1 = imread('processedSegment_n_100.jpg');
%C2 = imread('processedPhi_n_100.jpg');
%figure(),subplot(1,2,1),imshow(C1),title('Processed segment'),
%subplot(1,2,2),imshow(C2),title('Processed phi');
toc;
%% Comments
% The coded hard k-means algorithm starts with evaluating the distances
% between clusters and sample, samples with minimum distances are chosen and assigned to a
% cluster. Then the assigned clusters are updated by finding the mean value
% of corresponding rows from the data matrix, this step is followed by
% evaluating the distances again .. This process can be iterated 'Niter'(variable
% to defined by user) times. 
% To expand the comparison, the MATLAB function k-means is also applied. 
% kmeans by MATLAB uses a two-phase iterative algorithm to minimize the sum of point-to-centroid distances,
% summed over all k clusters. This first phase uses batch updates, where each iteration consists of reassigning
% points to their nearest cluster centroid, all at once, followed by recalculation of cluster centroids.
% The second phase uses online updates, where points are individually reassigned if doing
% so reduces the sum of distances, and cluster centroids are recomputed after each reassignment.
% For the ActiveContours App, 130 iterations used with '2' image update rate.
% It is possible to observe different color zones from the processed
% images, since the image is classified as melanoma, with the process we
% can confirm the diagnosis. If wanted the algorithm can be applied to
% different images from dataset.
