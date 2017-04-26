% % ICT4HEALTH LAB.5
% % ANI DEVER s225055
clear variables; close all; clc,tic;
%% preparing the data 
load('chronickidneydisease.mat');
y = chronickidneydisease;
keylist = {'normal','abnormal','present','notpresent','yes','no','good','poor','ckd','notckd','?',''};
keymap = [0,1,0,1,0,1,0,1,2,1,NaN,NaN]; % it is better not to have NaN
[M, N] = size(y);
for kr = 1:M
    for kc = 1:N
        c = strtrim(y{kr,kc}); % removes blanks
        check=strcmp(c,keylist); % check(i)=1 if c==keylist(i)
        if sum(check) == 0
            b(kr,kc) = str2num(y{kr,kc}); % from text to numeric
        else
            ii = find(check==1);
            b(kr,kc) = keymap(ii); % use the lists
        end
    end
end
% now b is the numerical data matrix converted from cell array of nominal
% values
class_id = b(:,end); % sparing the doctor's classification
iii = find(class_id==1); % index of healthy patients
y = b(:,1:(end-1)); % removing the doctor's classification 
%% Clustering
% Hierarchical clustering
euclidean = pdist(y); % computes the Euclidean distance between pairs of objects
Z1 = linkage(euclidean);
% matrix Z1 encodes a tree of hierarchical clusters of the rows of the real matrix
T = cluster(Z1,'maxclust',2);
% T constructs a maximum of 2 clusters using the 'distance' criterion. 
% Hierarchical tree generation
figure(),dendrogram(Z1,0), title('dendrogram with Euclidean distance'), grid on,
nn=1:M;
figure(),plot(nn,sortrows(T),'bo'), xlabel('i'), ylabel('cluster'),hold on,
plot(nn,sortrows(class_id),'rx'), title('Algorithm vs Doctors classification'), grid on;
% probabilities
err = immse(T,class_id);
fprintf('According to medical doctors =>\n');
fprintf('Total no of healthy patients: %i\n',length(iii));
fprintf('Total no of diseased patients: %i\n',length(class_id)-length(iii));
fprintf('According to algorithm =>\n');
fprintf('Total no of estimated healthy patients: %i\n',sum(T==1));
fprintf('Total no of estimated kidney diseased patients: %i\n',sum(T==2));
fprintf('MSE of classification is : %f\n', err);
false_positive = sum((T==2)&(class_id==1))/length(iii);
fprintf('Probability of false positive = %f\n', false_positive);
true_positive = sum((T==2)&(class_id==2))/(length(class_id)-length(iii));
fprintf('Probability of true positive = %f\n', true_positive);
false_negative = sum((T==1)&(class_id==2))/(length(class_id)-length(iii));
fprintf('Probability of false negative = %f\n', false_negative);
true_negative = sum((T==1)&(class_id==1))/length(iii);
fprintf('Probability of true negative = %f\n', true_negative);
%% Classification
% decision tree, doctor's classification is included 
bmean = nanmean(b); % mean of columns ignoring NaN's
% Here I have decided to convert the NaN's into column's mean value 
for i=1:M
    indx = isnan(b(i,:));
    b(i,indx) = bmean(indx);
end
% PCA
[M, N] = size(b);
b = b-ones(M,1)*mean(b);
R = b'*b/N;
[U,D] = eig(R);
Y = b*U*sqrt(inv(D));
tc = fitctree(Y,class_id);
view(tc, 'Mode', 'graph');
% Implementing the decision as specified by the decision tree
class_id_2 = ones(M,1);
% right branches
%class_id_2((Y(:,10)>=-0.0447061)&(Y(:,16)<0.326301))=1;
class_id_2((Y(:,10)>=-0.0447061)&(Y(:,16)<0.326301))=2;
% leftmost branch
class_id_2((Y(:,10)<-0.0447061)&(Y(:,10)<-0.146413))=2;
% 
class_id_2((Y(:,10)>=-0.0447061)&(Y(:,5)>=0.0605691))=2;
class_id_2((Y(:,10)>=-0.0447061)&(Y(:,5)<0.0605691)&(Y(:,23)<-0.0368987))=2;
% 
class_id_2((Y(:,10)>=-0.0447061)&(Y(:,5)<0.0605691)&(Y(:,23)>=-0.0368987)&(Y(:,11)<-0.32799))=2;
%class_id_2(class_id_2==0)=1;
% Implementing the decision as specified by the decision tree 
% with keeping NaNs and without applying PCA
% tcc = fitctree(y,class_id);
% class_id_3 = ones(M,1);
% class_id_3((y(:,15)<13.05)&(y(:,16)<44.5))=2;
% class_id_3((y(:,15)>=13.05)&(y(:,3)<1.0175))=2;
% class_id_3((y(:,15)>=13.05)&(y(:,3)<1.0175)&(y(:,4)>=0.5))=2;
% err3 = immse(class_id,class_id_3);
% fprintf('MSE of decision tree clustering is : %f\n', err3);
fprintf('According to hierarchical classification =>\n');
fprintf('Total no of estimated healthy patients: %i\n',sum(class_id_2==1));
fprintf('Total no of estimated kidney diseased patients: %i\n',sum(class_id_2==2));
err2 = immse(class_id,class_id_2);
fprintf('MSE of decision tree classification is : %f\n', err2);
false_positive = sum((class_id_2==2)&(class_id==1))/length(iii);
fprintf('Probability of false positive = %f\n', false_positive);
true_positive = sum((class_id_2==2)&(class_id==2))/(length(class_id)-length(iii));
fprintf('Probability of true positive = %f\n', true_positive);
false_negative = sum((class_id_2==1)&(class_id==2))/(length(class_id)-length(iii));
fprintf('Probability of false negative = %f\n', false_negative);
true_negative = sum((class_id_2==1)&(class_id==1))/length(iii);
fprintf('Probability of true negative = %f\n', true_negative);
figure(),plot(nn,sortrows(class_id),'bo'), xlabel('i'), ylabel('cluster'),hold on,
plot(nn,sortrows(class_id_2),'rx'), title('Classification tree vs Doctors classification'), grid on;
toc;
%% Comments
% In classification we are interested on the point x_k which
% represents region Rk , and we substitute y(n) with x_k if
% y(n) ∈ R k. In clustering we are just interested in the index of the region, and
% we simply state that y(n) belongs to cluster k if y(n) ∈ Rk.
% Preprocessing is applied to obtain a numerical data matrix. Embedded
% Matlab functions are used(i.e pdist, linkage, cluster) in order to
% perform clustering. The patients with NaN attributes are not considered by
% Matlab for the generation of the tree and they are therefore set aside
% in the dendrogram.
% Classification is performed by Matlab function fitctree(doctors class' are included).
% There were 2 possible ways to implement the decision specified by
% decision tree : with applying PCA and converting NaNs into the mean
% value of column, without applying PCA and keeping NaNs. Eventually I have
% observed that the first method(with PCA) gives me better results,
% therefore I have decided to include it, so the code lines of other
% approach is commented. 
% However if PCA is applied we no longer have the exact information of
% features. Therefore if an observation must be done using the decision
% tree, the second approach will be a wise choice.
% Personal observations : 
% Even before applying anykind of algorithm, I have seen that none of the healthy
% patients have 'anemia','pedal edema','hypertension','diabetes mellitus',
% 'coronary artery disease' attributes as 'yes', they are either 'no' or ?.
% So these features can be interesting to observe. Also the average age of diseased patients are
% higher than healthy patients, but that is not a exact discriminator
% criteria since there is a diseased patient who is 7 years old. 
% Then the decision tree picks 15th, 16th features (sodium and potassium respectively)
% as a starting point then followed by the 3rd feature 'Specific Gravity'.
% When compared the results of 2 different approaches, it is possible to
% say that hierarchical classification performs better, even if the
% estimated number of patients(healthy, diseased) are relatively poor but what
% really important is the sensitivity and specificity. 

