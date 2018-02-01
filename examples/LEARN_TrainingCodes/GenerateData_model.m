
%%% Generate the training data.

clear all;close all;

addpath(genpath('./.'));
%addpath utilities;

batchSize      = 1;        %%% batch size
max_numPatches = batchSize*2000; 
modelName      = 'LEARN';
sigma          = 25;         %%% Gaussian noise level

%%% training and testing
load traindata.mat;
load proMatrix_64.mat;

val_train     = 0;           %%% training % default
val_test      = 1;           %%% testing  % default
f = inputf(:,1:200);%f
sr = input(:,:,1:200); %u
im = label(:,:,1:200); % ori

f1 = inputf(:,201:225);%f
sr1 = input(:,:,201:225)); %u
im1 = label(:,:,201:225); % ori
%%% training patches
[inputs, inputsf,labels, set]  = patches_generation(sr,im,val_train,max_numPatches,batchSize,f);
%%% testing  patches
[inputs2,inputsf2,labels2,set2] = patches_generation(sr1,im1,val_test,max_numPatches,batchSize,f1);

inputs   = cat(4,inputs,inputs2);      clear inputs2;
labels   = cat(4,labels,labels2);      clear labels2;
inputsf  = cat(3,inputsf,inputsf2);    clear inputsf2;
set      = cat(2,set,set2);            clear set2;

if ~exist(modelName,'file')
    mkdir(modelName);
end

%%% save data
save(fullfile('imdbtrain'), 'inputs','labels','set','inputsf','-v7.3')

