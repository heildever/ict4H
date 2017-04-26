% % ICT4HEALTH LAB.10
% % ANI DEVER s225055
clear variables; close all; clc, tic;
%% defining variables and voice recording
% Fsamp = 8e3; % sampling frequency
Nbits = 8; % no of quantization bits
Nstates = 8;
% Nchann = 1; % no of channels
% interval = 1; % voice recording interval 1sec
% for i=1:5
%     recObj = audiorecorder(Fsamp, Nbits, Nchann);
%     disp('Start speaking after hitting the key');
%     st_sig = input('\nHit any key to continue');
%     recordblocking(recObj, interval);
%     myRecording(:,i) = getaudiodata(recObj);
% end
% figure(),plot(myRecording),grid on; % plotting the recorded samples
%% Quantization
Kquant = 16;
% for i=1:5
%     amax(i) = max(myRecording(:,i));
%     amin(i) = min(myRecording(:,i));
%     delta(i) = (amax(i)-amin(i))/(Kquant-1); % quantization interval
%     ar(:,i) = round((myRecording(:,i)-amin(i))/delta(i))+1; % quantized signal
%     each column of 'ar' matrix represents the quantized signal of
%     the recorded vowel 
% end
% figure(),plot(ar),grid on; % plotting the quantized samples
%% HMM training
rng('default');
TRANS_HAT = rand(Nstates,Nstates); % initial transition matrix of size 8x8  
EMIT_HAT = rand(Nstates,Kquant); % state transition matrix - randomly initialized
% Normalizing in order to get 1 as sum of the row
dummy = sum(TRANS_HAT,2);
for i=1:8 
    TRANS_HAT(i,:) = TRANS_HAT(i,:)/dummy(i);
end
dummy = sum(EMIT_HAT,2);
for i=1:8
    EMIT_HAT(i,:) = EMIT_HAT(i,:)/dummy(i);
end
% load('Lab10_Health/recording_A');
% [ESTTRa,ESTEMITa] = hmmtrain(recording_A,TRANS_HAT,EMIT_HAT,'Maxiteration',400);
% save('Lab10_Health/HMMa','ESTTRa','ESTEMITa'); % each HMM is trained and saved

% load('Lab10_Health/recording_E');
% [ESTTRe,ESTEMITe] = hmmtrain(recording_E,TRANS_HAT,EMIT_HAT,'Maxiteration',400);
% save('Lab10_Health/HMMe','ESTTRe','ESTEMITe'); % each HMM is trained and saved

% load('Lab10_Health/recording_I');
% [ESTTRi,ESTEMITi] = hmmtrain(recording_I,TRANS_HAT,EMIT_HAT,'Maxiteration',400);
% save('Lab10_Health/HMMi','ESTTRi','ESTEMITi'); % each HMM is trained and saved

% load('Lab10_Health/recording_O');
% [ESTTRo,ESTEMITo] = hmmtrain(recording_O,TRANS_HAT,EMIT_HAT,'Maxiteration',400);
% save('Lab10_Health/HMMo','ESTTRo','ESTEMITo'); % each HMM is trained and saved

% load('Lab10_Health/recording_U');
% [ESTTRu,ESTEMITu] = hmmtrain(recording_U,TRANS_HAT,EMIT_HAT,'Maxiteration',400);
% save('Lab10_Health/HMMu','ESTTRu','ESTEMITu'); % each HMM is trained and saved

% loading pre-trained HMM's
load('Lab10_Health/HMMa');
load('Lab10_Health/HMMe');
load('Lab10_Health/HMMi');
load('Lab10_Health/HMMo');
load('Lab10_Health/HMMu');
load('Lab10_Health/zc'); % zc is constructed by recordings itself
%% Recognition 
Prob = zeros(5,5);
for i=1:5
    ta = (zc(:,i)).'; % at each iteration 'ta' becomes the sample of a,e,i,o,u respectively
    [PSTATESa,logpseqa] = hmmdecode(ta,ESTTRa,ESTEMITa);
    [PSTATESe,logpseqe] = hmmdecode(ta,ESTTRe,ESTEMITe);
    [PSTATESi,logpseqi] = hmmdecode(ta,ESTTRi,ESTEMITi);
    [PSTATESo,logpseqo] = hmmdecode(ta,ESTTRo,ESTEMITo);
    [PSTATESu,logpsequ] = hmmdecode(ta,ESTTRu,ESTEMITu);
    % exp(logpseqa) produces 0 everytime
    Prob(i,1:5)=[logpseqa, logpseqe, logpseqi, logpseqo, logpsequ];
end
disp(Prob);
toc;
%% Comments
% I have tried different voice recordings in order to have reasonable
% samples to train. I started to pronounce the vowel then started
% recording, I have used pronounciations provided by professionals. 
% For this execution I am placing my own voice sample which I have started
% to pronounce before starting recording(I tried to avoid useless bits in the sample)
% I have trained the Markov models with 400 iterations which took more than 4000seconds
% per each.(around 4028 secs)
% The trained emission and transition matrices are then 
% loaded again in the script. Recognition has been done and the logpseq, 
% (the logarithm of the probability of the test sequence) values are stored
% in a matrix for each vowel. 
% However, the detection does not produce the desired output. 
