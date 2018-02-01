
%%% Note: run the 'GenerateData_model_64_25_Res_Bnorm_Adam.m' to generate
%%% training data first.

clear all;

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

global ASRmtx  ASRmtx_T
load proMatrix_64.mat;
ASRmtx = systemMatrix;
ASRmtx_T = ASRmtx';
clear systemMatrix;
%%%-------------------------------------------------------------------------
%%% configuration
%%%-------------------------------------------------------------------------
opts.modelName        = 'LEARN_Model'; %%% model name
opts.learningRate     = 0.0001;%%% you can change the learning rate
opts.batchSize        = 1; %%% default
opts.gpus             = 1; %%% this code can only support one GPU!
%%% solver
opts.solver           = 'Adam';
opts.gradientClipping = false; %%% Set 'true' to prevent exploding gradients in the beginning.
opts.expDir      = fullfile('data', opts.modelName);
opts.imdbPath    = fullfile(opts.expDir, 'imdbtrain.mat');

%%%-------------------------------------------------------------------------
%%%   Initialize model and load data
%%%-------------------------------------------------------------------------
%%%  model
net  = feval(['Init_',opts.modelName]);

%%%  load data
imdb = load(opts.imdbPath) ;

%%%-------------------------------------------------------------------------
%%%   Train 
%%%-------------------------------------------------------------------------

[net, info] = CNN_train(net, imdb, ...
    'expDir', opts.expDir, ...
    'learningRate',opts.learningRate, ...
    'solver',opts.solver, ...
    'gradientClipping',opts.gradientClipping, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus) ;






