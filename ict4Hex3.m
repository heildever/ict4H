% % ICT4HEALTH LAB.3
% % ANI DEVER s225055
clear variables, close all; clc,tic;
%% preprocessing the data 
load('arrhythmia.mat');
x = arrhythmia; % naming the matrix as x, to ease the usage
class_id = x(:,end); % vector of classes(1,2), last column of data matrix
class_id2 = x(:,end);
iii = find(class_id==1);
jjj = find(class_id>1); 
class_id(jjj)=2; % any number > 1, transform it to 2
N1 = length(iii); % no of class1 patients
N2 = length(jjj); % no of class2 patients
fprintf('According to doctor diagnosis =>\n');
fprintf('Total no of healthy patients: %i\n',N1);
fprintf('Total no of arrhythmic patients: %i\n',N2);
x = x(:,any(x)); % removing the columns with only 0's
y = x(1:end,1:end-1); % y is a matrix without the class data, removing last column 
y1 = y(iii,:); % class_1 healthy 
y2 = y(jjj,:); % class_2 arrhythmic
x1 = mean(y1); % the mean of the column vectors in y1
x2 = mean(y2); % the mean of the column vectors in y2
[M,N] = size(y);
o = ones(M,1);
%% Minimum distance criterion
xmeans = [x1;x2];% matrix with x1 and x2
eny = diag(y*y');% |y(n)|^2
enx = diag(xmeans*xmeans');% |x1|^2 and |x2|^2
dotprod = y*xmeans';% matrix with the dot product between each y(n) and each x
[U, V] = meshgrid(enx,eny);
dist2 = U+V-2*dotprod;%|y(n)|^2+|x(k)|^2-2y(n)x(k)= =|y(n)-x(k)|^2
% estimating the classes according to minimum distance criterion
[~,est_class_id]=min(dist2,[],2);
pi1=sum(est_class_id==1);
pi2=sum(est_class_id==2);
% probabilities
fprintf('According to minimum distance criterion =>\n');
fprintf('Total no of estimated healthy patients: %i\n',pi1);
fprintf('Total no of estimated arrhythmic patients: %i\n',pi2);
false_positive = sum((est_class_id==2)&(class_id==1))/N1;
fprintf('Probability of false positive = %f\n', false_positive);
true_positive = sum((est_class_id==2)&(class_id==2))/N2;
fprintf('Probability of true positive = %f\n', true_positive);
false_negative = sum((est_class_id==1)&(class_id==2))/N2;
fprintf('Probability of false negative = %f\n', false_negative);
true_negative = sum((est_class_id==1)&(class_id==1))/N1;
fprintf('Probability of true negative = %f\n', true_negative);
error_mdc = immse(est_class_id,class_id);
fprintf('The msee of min distance criterion is : %f\n',error_mdc);
figure(), plot(sortrows(class_id),'bo'),hold on,plot(sortrows(est_class_id),'r*'),
grid on,title('Minimum distance criterion');
%% PCA
y = normalize(y);
R=y'*y/N;[U,D]=eig(R);
d=diag(D);d1=d/sum(d);d1c=cumsum(d1);
% keeping the significant eigenvalues
removed_eigen=5e-3;nrem=(d1c<removed_eigen);
UL=U;UL(:,nrem)=[];
fprintf('%i eigenvalues(<%f) out of %i are removed\n',sum(nrem==1),removed_eigen,N);
z=y*UL; z=z./(o*sqrt(var(z)));
% minimum distance criterion again
z1=z(iii,:);z2=z(jjj,:); 
w1=mean(z1);w2=mean(z2); wmeans=[w1;w2]; rhoz=z*wmeans';
en1=diag(z*z');en2=diag(wmeans*wmeans'); [Uy,Vy]=meshgrid(en2,en1);
distz=Uy+Vy-2*rhoz;
[~,decz]=min(distz,[],2);
% probabilities
fprintf('According to PCA criterion =>\n');
fprintf('Total no of estimated healthy patients: %i\n',sum(decz==1));
fprintf('Total no of estimated arrhythmic patients: %i\n',sum(decz==2));
false_positive_PCA=sum((decz==2)&(class_id==1))/N1;
fprintf('Probability of false positive = %f\n', false_positive_PCA);
true_positive_PCA=sum((decz==2)&(class_id==2))/N2;
fprintf('Probability of true positive = %f\n', true_positive_PCA);
false_negative_PCA=sum((decz==1)&(class_id==2))/N2;
fprintf('Probability of false negative = %f\n', false_negative_PCA);
true_negative_PCA=sum((decz==1)&(class_id==1))/N1;
fprintf('Probability of true negative = %f\n', true_negative_PCA);
error_pca = immse(decz,class_id);
fprintf('The msee of PCA is : %f\n',error_pca);
figure(), plot(sortrows(class_id),'bo'),hold on,plot(sortrows(decz),'r*'),
grid on,title('PCA');
%% Bayesian approach
pis=zeros(1,2);
pis(1)=N1/N;pis(2)=N2/N;
dist2b=distz-2*o*log(pis); % from the square distance we remove 2*sig2*log(pi)
[~,decb]=min(dist2b,[],2);
% probabilities
fprintf('According to Bayesian criterion =>\n');
fprintf('Total no of estimated healthy patients: %i\n',sum(decz==1));
fprintf('Total no of estimated arrhythmic patients: %i\n',sum(decz==2));
false_positive_B=sum((decb==2)&(class_id==1))/N1;
fprintf('Probability of false positive = %f\n', false_positive_B);
true_positive_B=sum((decb==2)&(class_id==2))/N2;
fprintf('Probability of true positive = %f\n', true_positive_B);
false_negative_B=sum((decb==1)&(class_id==2))/N2;
fprintf('Probability of false negative = %f\n', false_negative_B);
true_negative_B=sum((decb==1)&(class_id==1))/N1;
fprintf('Probability of true negative = %f\n', true_negative_B);
error_b = immse(decb,class_id);
fprintf('The msee of Bayes criterion is : %f\n',error_b);
figure(), plot(sortrows(class_id),'bo'),hold on,plot(sortrows(decb),'r*'),
grid on,title('Bayes criterion');
% Bayesian continued
[N1,F1]=size(z1); dd1=z1-ones(N1,1)*w1; R1=dd1'*dd1/N1; R1=inv(R1);
[N2,F2]=size(z2); dd2=z2-ones(N2,1)*w2; R2=dd2'*dd2/N2; R2=inv(R2);
G=zeros(M,2);
for n=1:M
    G(n,1)=(z(n,:)-w1)*R1*(z(n,:)-w1)'+log(det(R1))-2*log(pis(1));
    G(n,2)=(z(n,:)-w2)*R2*(z(n,:)-w2)'+log(det(R2))-2*log(pis(2));
end
[a,decb2]=min(G,[],2);
% probabilities
fprintf('According to second approach of Bayesian criterion =>\n');
fprintf('Total no of estimated healthy patients: %i\n',sum(decb2==1));
fprintf('Total no of estimated arrhythmic patients: %i\n',sum(decb2==2));
false_positive_B=sum((decb2==2)&(class_id==1))/N1;
fprintf('Probability of false positive = %f\n', false_positive_B);
true_positive_B=sum((decb2==2)&(class_id==2))/N2;
fprintf('Probability of true positive = %f\n', true_positive_B);
false_negative_B=sum((decb2==1)&(class_id==2))/N2;
fprintf('Probability of false negative = %f\n', false_negative_B);
true_negative_B=sum((decb2==1)&(class_id==1))/N1;
fprintf('Probability of true negative = %f\n', true_negative_B);
error_b = immse(decb2,class_id);
fprintf('The msee of Bayes criterion is : %f\n',error_b);
figure(), plot(sortrows(class_id),'bo'),hold on,plot(sortrows(decb2),'r*'),
grid on,title('Bayes criterion');
toc;
%% Comments
% Before applying any kind of algorithm, necessary variables were extracted
% and preprocessing is applied.
% The idea behind Minimum distance criterion is to assign the vector of y to
% Rk(region K) if that vector of y is closer to Xk than any other representative point.
% Square distances are measured at the related section and assignments are done.
% PCA is included in order to improve the results. 
% Then for Bayesian criterion the steps are followed as stated in the lab
% assignment.
% In particular, the script provided by the professor is taken as a reference.
% At the end it possible to examine different approaches' performances.
% According the probabilities it can be said that Bayes criterion performs
% best and it is followed by PCA and Minimum distance criterion.
% Then after Bayes is continued with further steps, which will lead us to
% perfect results. 
% As a reminder the plots are not an exact indicator of results,
% since they are sorted it shows the number of estimates. Even if
% the number of estimated values match with doctors', doesnt mean that
% results are perfect, individual conclusions are important; i.e the case
% of 88th patient.
% It is better to observe mean square errors for precision.
