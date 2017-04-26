% % ICT4HEALTH LAB.2
% % ANI DEVER s225055
clear variables; close all; clc,tic;
%% preparing the data 
load('parkinsonnew.mat'); % sorted matrix 
data_train = parkinsonnew(1:840,1:end); % train matrix
data_test = parkinsonnew(841:end,1:end); % test matrix
data_train_norm = data_train(1:840,1:6);
data_test_norm = data_test(1:150,1:6);
%% means and st devs of the matrices
m_data_train = mean(data_train,1);
v_data_train = std(data_train,1);
F0 = 5;
fprintf('F0 value is : %i\n',F0);
% Normalization of train data matrix
for i = 1:840
    for l = F0:22 
        data_train_norm(i,l) = (data_train(i,l)-m_data_train(l))./v_data_train(l);
    end
end
% Normalization of test data matrix
for i = 1:150
    for l = F0:22
        data_test_norm(i,l) = (data_test(i,l)-m_data_train(l))./v_data_train(l);
    end    
end
% 
y_train = data_train_norm(:,F0);
X_train = data_train_norm;
X_train(:,F0) = [];
y_test = data_test_norm(:,F0);
X_test = data_test_norm;
X_test(:,F0)=[];
%% Linear regression techniques
%% MSE
[yhat_mse_train, ahat_mse_train] = MSE(X_train,y_train);
[yhat_mse_test, ahat_mse_test] = MSE(X_test,y_test);
error_mse = immse(yhat_mse_train,y_train);
fprintf('The msee of MSE is : %f\n',error_mse);
figure(),subplot(1,2,1), plot(ahat_mse_train), title('a hat for train'), grid on,
subplot(1,2,2), plot(ahat_mse_test), grid on,title('a hat for test'), suptitle('a hat values for MSE');
figure(),subplot(1,2,1),plot(yhat_mse_train,'m-.','LineWidth',2),ylabel('values'),
hold on, plot(y_train),xlabel('y train / yhat train'),grid on,
subplot(1,2,2),plot(yhat_mse_test,'r--o','LineWidth',2),ylabel('values'),hold on,
plot(y_test), xlabel('y test / yhat test'),grid on,suptitle('MSE estimates');
%% Gradient algorithm
gama = 1e-8; % learning coefficient
[yhat_grd_train, ahat_grd_train] = grad(X_train, y_train, gama);
[yhat_grd_test, ahat_grd_test] = grad(X_test, y_test, gama);
error_grd = immse(yhat_grd_train,y_train);
fprintf('The msee of gradient algorithm is : %f\n',error_grd);
figure(),subplot(1,2,1), plot(ahat_grd_train), title('a hat for train'), grid on,
subplot(1,2,2), plot(ahat_grd_test), grid on, title('a hat for test'),
suptitle('a hat values for gradient algorithm'); 
figure(),subplot(1,2,1),plot(yhat_grd_train,'m-.','LineWidth',2),ylabel('values'),
hold on, plot(y_train),xlabel('y train / yhat train'),grid on;
subplot(1,2,2),plot(yhat_grd_test,'r--o','LineWidth',2),ylabel('values'),hold on,
plot(y_test), xlabel('y test / yhat test'),grid on,suptitle('Gradient algorithm estimates');
%% Steepest descent algorithm
[yhat_sd_train, ahat_sd_train]= stedes(X_train, y_train); % estimate of y
[yhat_sd_test, ahat_sd_test] = stedes(X_test, y_test);
error_sd = immse(yhat_sd_train,y_train);
fprintf('The msee of steepest descent algorithm is : %f\n',error_sd);
figure(),subplot(1,2,1), plot(ahat_sd_train), title('a hat for train'), grid on,
subplot(1,2,2), plot(ahat_sd_test), grid on, title('a hat for test'),
suptitle('a hat values for steepest descent algorithm');
figure(),subplot(1,2,1),plot(yhat_sd_train,'m-.','LineWidth',2),ylabel('values'),
hold on, plot(y_train),xlabel('y train / yhat train'),grid on,
subplot(1,2,2),plot(yhat_sd_test,'r--o','LineWidth',2),ylabel('values'),hold on,
plot(y_test), xlabel('y test / yhat test'),grid on,suptitle('Steepest descent algorithm estimates');
%% PCA
Z_train = PCA(X_train);
Z_test = PCA(X_test);
%% PCR 
[est_train, ahat_pcr_train] = PCR(X_train, y_train, Z_train);
[est_test, ahat_pcr_test] = PCR(X_test, y_test, Z_test);
error_pcr = immse(est_train,y_train);
fprintf('The msee of PCR algorithm is : %f\n',error_pcr);
figure(),subplot(1,2,1),plot(est_train,'m-.','LineWidth',2),ylabel('values'),
hold on, plot(y_train),xlabel('y train / yhat train'),grid on,
subplot(1,2,2),plot(est_test,'r--o','LineWidth',2),ylabel('values'),hold on,
plot(y_test), xlabel('y test / yhat test'),grid on,suptitle('PCR estimates');
%
figure(),subplot(1,2,1),histogram(y_train-est_train,50),title('y train - yhat train'),grid on;
subplot(1,2,2),histogram(y_test-est_test,50),title('y test - yhat test'),grid on,suptitle('Histograms');
% PCR with reduced features
[est_red_train, ahat_pcrR_train] = PCR_red(X_train, y_train);
[est_red_test, ahat_pcrR_test] = PCR_red(X_test, y_test);
%
figure(),subplot(1,2,1),plot(ahat_pcr_train,'DisplayName','non reduced'),hold on,
plot(ahat_pcrR_train,'m--','LineWidth',2,'DisplayName','reduced'),title('a hat for train'),
grid on, subplot(1,2,2),plot(ahat_pcr_test,'DisplayName','non reduced'),hold on,
plot(ahat_pcrR_test,'m--','LineWidth',2,'DisplayName','reduced'),title('a hat for test'),
grid on,legend('show');
%
figure(),subplot(1,2,1),plot(est_red_train,'m-.','LineWidth',2),ylabel('values'),
hold on, plot(y_train),xlabel('y train / yhat train'),grid on,
subplot(1,2,2),plot(est_red_test,'r--o','LineWidth',2),ylabel('values'),hold on,
plot(y_test), xlabel('y train / yhat test'),grid on,suptitle('PCR estimates with reduced features');
%
error_red_pcr = immse(est_red_train,y_train);
fprintf('The msee of PCR algorithm with reduced features is : %f\n',error_red_pcr);
figure(),subplot(1,2,1),histogram(y_train-est_red_train,50),title('y train - yhat train **REDUCED'),
grid on, subplot(1,2,2),histogram(y_test-est_red_test,50),title('y test - yhat test **REDUCED'),
suptitle('Histograms with reduced features'),grid on;
toc;
%% Comments
% This report is written and compiled while F0=5.
% First of all preprocessing(normalization, extraction of train and test values) is applied.
% Different linear regression techniques are studied, applied.
% The plots and printed outputs are available in order to compare the
% results. In order to provide a readable report, these algorithms are written
% as Matlab functions and called when necessary. 
% The MSE solution requires that a matrix is inverted, which might be
% too complex is some cases. Then gradient iterative solution can be used.
% The selection of learning coefficient of gradient algorithm and stoping threshold affects the
% results and execution time significantly. The correct value of γ depends on the specific function to be
% minimized. A faster convergence can be obtained with the steepest descent
% algorithm, which finds the “optimum” value of γ at each step.
% The experiment is then continued with PCA to obtain Z. 
% The techniques(both PCR and PCR with reduced features) are executed as
% described in related course material. Despite the fact that MSE is
% slightly simpler than others the performance of the algorithm is
% promising. In fact the msee's of MSE and PCR is the same. Then they are
% followed by steepest descent and gradient algorithm. 
