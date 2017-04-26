% % ICT4HEALTH LAB.9
% % ANI DEVER s225055
clear variables; close all; clc,tic;
%% Preprocessing
% load the arrhytmia matrix
load('arrhythmia.mat');
x = arrhythmia; % data matrix
% modifying the classes as 1s and -1s
iii = find(x(:,end)>1); x(iii,end)=-1;
% removing columns with only 0s
x = x(:,any(x));
class_id = x(:,end); % vector of classes
x = x(1:end,1:end-1); % removing the class info from the data matrix
[rows, cols] = size(x);
x = normalize(x);
N = 200 ; % no of Boxconstraints to try
%% Preallocation of cell arrays 
% to decrease execution time
Mdl_linear = cell(1,N);
Mdl_gaussian = cell(1,N);
CVMdl_linear = cell(1,N);
CVMdl_gaussian = cell(1,N);
classLoss_linear = cell(1,N);
classLoss_gaussian = cell(1,N);
%% SVM
rng('default');
for i=1:N 
    Mdl_linear{i} = fitcsvm(x,class_id,'BoxConstraint',i,'KernelFunction','linear');
    Mdl_gaussian{i} = fitcsvm(x,class_id,'BoxConstraint',i,'KernelFunction','gaussian'); 
    CVMdl_linear{i} = crossval(Mdl_linear{i});
    CVMdl_gaussian{i} = crossval(Mdl_gaussian{i});
    classLoss_linear{i} = kfoldLoss(CVMdl_linear{i});
    classLoss_gaussian{i} = kfoldLoss(CVMdl_gaussian{i});
end
min_linear = min(cell2mat(classLoss_linear));
min_loss_linear = find(cell2mat(classLoss_linear)==min_linear);
min_gaussian = min(cell2mat(classLoss_gaussian));
min_loss_gaussian = find(cell2mat(classLoss_gaussian)==min_gaussian);
classhat_linear = sign(x*Mdl_linear{min_loss_linear}.Beta+Mdl_linear{min_loss_linear}.Bias);
% Mdl.Beta = [] for gaussian
% classhat_gaussian = sign(y+Mdl_gaussian.Bias);   
fprintf('Linear class-loss is : %d\n',min_linear);
fprintf('Gaussian class-loss is : %d\n', min_gaussian);
figure, plot(sortrows(class_id),'bo'),hold on,plot(sortrows(classhat_linear),'r*'),
grid on,title('Linear kernel classification');
toc;
%% Comments
% We assume that class info is either 1 or -1, so 2 hyperspaces are
% separated by the hyperplanes(with given equations). We prefer the hyperplane that
% is at “maximum distance from the closest points”, because then we have a larger
% margin for the correct classification of new patiens. The closest points
% are called support vectors. The MATLAB function 'fitcsvm' returns a support vector
% machine classifier Mdl trained using the sample data.
% The models are trained in a loop and stored in a cell array in order to see the
% effect of using different BoxConstraint (the number of iterations can be specified by changing
% the variable 'N' here I have chosen 200, the number of iterations affects
% the execution time majorly, however a 'BoxConstraint' value must
% be an integer). After the traning of models, misclasifications are evaluated
% by kfoldLoss command of MATLAB which basically returns the
% cross-validation loss(a mean misclassification probability) of the cvmodels.
% The models with minimum loss' are chosen to estimate the classes. The linear kernel
% function enables us to visualize the estimate classes, therefore the plot is
% referring to the comparison between estimates and classification done by doctors. 
