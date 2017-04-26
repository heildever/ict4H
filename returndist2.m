function [ dist2 ] = returndist2()
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
load('arrhythmia.mat');
N = length(arrhythmia);
for i=1:N
    if arrhythmia(i,275) == 1 % leave as '1' if it is already '1'
       continue
    else
        arrhythmia(i,275) = 2; % make it '2' if it is greater than '1'
    end
end
arrhythmia = arrhythmia(:,any(arrhythmia)); % removing the columns with only 0's
class_id = arrhythmia(:,258); % vector of classes(1,2) in given order
y = arrhythmia(1:end,1:257); % y is a matrix without the class data 
arrhythmia = sortrows(arrhythmia,258); % sorting according to class_id at last column
y1 = y(1:245,1:end); % class_1 healthy 
y2 = y(246:end,1:end); % class_2 arrhythmic
x1 = mean(y1,1); % the mean of the column vectors in y1
x2 = mean(y2,1); % the mean of the column vectors in y2
%% Minimum distance criterion
xmeans = [x1;x2];% matrix with x1 and x2
eny = diag(y*transpose(y));% |y(n)|^2
enx = diag(xmeans*transpose(xmeans));% |x1|^2 and |x2|^2
dotprod = y*transpose(xmeans);% matrix with the dot product between each y(n) and each x
[U, V] = meshgrid(enx,eny);
dist2 = U+V-2*dotprod;%|y(n)|^2+|x(k)|^2-2y(n)x(k)= =|y(n)-x(k)|^2
end

