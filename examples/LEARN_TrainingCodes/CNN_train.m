function [net, state] = CNN_train(net, imdb, varargin)


%%%-------------------------------------------------------------------------
%%% solvers: SGD(default) and Adam with(default)/without gradientClipping
%%%-------------------------------------------------------------------------
opts.solver           = 'Adam';
opts.beta1   = 0.9;
opts.beta2   = 0.999;
opts.alpha   = 0.01;
opts.epsilon = 1e-8;
opts.learningRate = 0.0001;
opts.weightDecay  = 0.0001;
opts.momentum     = 0.9 ;
opts.theta        = 0.005;

%%%-------------------------------------------------------------------------
%%%  setting for simplenn
%%%-------------------------------------------------------------------------

opts.conserveMemory = false ;
opts.mode = 'normal';
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false;

%%%-------------------------------------------------------------------------
%%%  setting for model
%%%-------------------------------------------------------------------------

opts.batchSize        = 2; 
opts.gpus             = 1; 
opts.gradientClipping = false; 
opts.numEpochs =100 ;
opts.modelName   = 'model';
opts.expDir = fullfile('data',opts.modelName) ;
opts.train = find(imdb.set==1);
opts.test  = find(imdb.set==2);

%%%-------------------------------------------------------------------------
%%%  update settings
%%%-------------------------------------------------------------------------

opts = vl_argparse(opts, varargin);
if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

%%%-------------------------------------------------------------------------
%%%  Initialization
%%%-------------------------------------------------------------------------

net = vl_simplenn_tidy(net);    %%% fill in some eventually missing values
net.layers{end-1}.precious = 1;
vl_simplenn_display(net, 'batchSize', opts.batchSize) ;

state.getBatch = getBatch ;

%%%-------------------------------------------------------------------------
%%%  Train and Test
%%%-------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf([opts.modelName,'-epoch-%d.mat'], ep));

 start = findLastCheckpoint(opts.expDir,opts.modelName) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d', mfilename, start) ;
     load(modelPath(start), 'net') ;
     net = vl_simplenn_tidy(net) ;
end

for epoch = 1 : opts.numEpochs
     %%% Train for one epoch.
    state.epoch = epoch ;
    state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate)));
    state.train = opts.train(randperm(numel(opts.train))) ; %%% shuffle
%     state.train = opts.train ; %%% shuffle
    state.test  = opts.test; %%% no need to shuffle
    opts.thetaCurrent = opts.theta(min(epoch, numel(opts.theta)));
    if numel(opts.gpus) == 1
        net = vl_simplenn_move(net, 'gpu') ;
    end
    
    [net, state] = process_epoch(net, state, imdb, opts, 'train');
%     [net,  ~   ] = process_epoch(net, state, imdb, opts, 'test' );    
    net = vl_simplenn_move(net, 'cpu');
    save(modelPath(epoch), 'net');    
end


%%%-------------------------------------------------------------------------
function  [net, state] = process_epoch(net, state, imdb, opts, mode)
%%%-------------------------------------------------------------------------

if strcmp(mode,'train')
    
    switch opts.solver
        
        case 'SGD' %%% solver: SGD
            for i = 1:numel(net.layers)
                if isfield(net.layers{i}, 'weights')
                    for j = 1:numel(net.layers{i}.weights)
                        state.layers{i}.momentum{j} = 0;
                    end
                end
            end
            
        case 'Adam' %%% solver: Adam
            for i = 1:numel(net.layers)
                if isfield(net.layers{i}, 'weights')
                    for j = 1:numel(net.layers{i}.weights)
                        state.layers{i}.t{j} = 0;
                        state.layers{i}.m{j} = 0;
                        state.layers{i}.v{j} = 0;
                    end
                end
            end
            
    end
    
end


subset = state.(mode) ;
num = 0 ;
res = [];
for t=1:opts.batchSize:numel(subset)
    
    %%% get this image batch
    batchStart = t;
    if strcmp(mode,'train')
    batchEnd = min(t+opts.batchSize-1, numel(subset));
    else
    batchEnd = min(t+16-1, numel(subset));
    end
    batch = subset(batchStart : 1: batchEnd);
    
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end
    
    [inputs,inputsf,labels] = state.getBatch(imdb, batch) ;
    
    if numel(opts.gpus) == 1
        inputs = gpuArray(inputs);
        labels = gpuArray(labels);
    end
    
    if strcmp(mode, 'train')
        dzdy = single(1);
        evalMode = 'normal';%%% forward and backward (Gradients)
    else
        dzdy = [] ;
        evalMode = 'test';  %%% forward only
    end
    
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, inputs,inputsf,dzdy, res, ...
        'mode', evalMode, ...
        'conserveMemory', opts.conserveMemory, ...
        'backPropDepth', opts.backPropDepth, ...
        'cudnn', opts.cudnn) ;
    
    if strcmp(mode, 'train')
        [state, net] = params_updates(state, net, res, opts, opts.batchSize) ;
    end
    
    lossL2 = gather(res(end).x) ;
    
    %%%--------add your code here------------------------
    
    %%%--------------------------------------------------
    
    fprintf('%s: epoch %02d: %3d/%3d:', mode, state.epoch, ...
        fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    fprintf('error: %f \n', lossL2) ;
    
end


%%%-------------------------------------------------------------------------
function [state, net] = params_updates(state, net, res, opts, batchSize)
%%%-------------------------------------------------------------------------

switch opts.solver
    
    case 'SGD' %%% solver: SGD
        
        for l=numel(net.layers):-1:1
            for j=1:numel(res(l).dzdw)
                if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
                    %%% special case for learning bnorm moments
                    thisLR = net.layers{l}.learningRate(j) - opts.bnormLearningRate;
                    net.layers{l}.weights{j} = ...
                        (1 - thisLR) * net.layers{l}.weights{j} + ...
                        (thisLR/batchSize) * res(l).dzdw{j} ;
                else
                    thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j);
                    thisLR = state.learningRate * net.layers{l}.learningRate(j);
                    
                    if opts.gradientClipping
                        theta = opts.thetaCurrent/thisLR;
                        state.layers{l}.momentum{j} = opts.momentum * state.layers{l}.momentum{j} ...
                            - thisDecay * net.layers{l}.weights{j} ...
                            - (1 / batchSize) * gradientClipping(res(l).dzdw{j},theta) ;
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
                            thisLR * state.layers{l}.momentum{j} ;
                    else
                        state.layers{l}.momentum{j} = opts.momentum * state.layers{l}.momentum{j} ...
                            - thisDecay * net.layers{l}.weights{j} ...
                            - (1 / batchSize) * res(l).dzdw{j} ;
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
                            thisLR * state.layers{l}.momentum{j} ;
                    end
                end
            end
        end
        
        
    case 'Adam'  %%% solver: Adam
        
        for l=numel(net.layers):-1:1
            for j=1:numel(res(l).dzdw)
                
                if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
                    %%% special case for learning bnorm moments
                    thisLR = net.layers{l}.learningRate(j) - opts.bnormLearningRate;
                    net.layers{l}.weights{j} = ...
                        (1 - thisLR) * net.layers{l}.weights{j} + ...
                        (thisLR/batchSize) * res(l).dzdw{j} ;
                else
                    thisLR = state.learningRate * net.layers{l}.learningRate(j);
                    state.layers{l}.t{j} = state.layers{l}.t{j} + 1;
                    t = state.layers{l}.t{j};
                    alpha = thisLR;
                    lr = alpha * sqrt(1 - opts.beta2^t) / (1 - opts.beta1^t);
                    
                    state.layers{l}.m{j} = state.layers{l}.m{j} + (1 - opts.beta1) .* (res(l).dzdw{j} - state.layers{l}.m{j});
                    state.layers{l}.v{j} = state.layers{l}.v{j} + (1 - opts.beta2) .* (res(l).dzdw{j} .* res(l).dzdw{j} - state.layers{l}.v{j});
                    
                    if opts.gradientClipping
                        theta = opts.thetaCurrent/lr;
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} - lr * gradientClipping(state.layers{l}.m{j} ./ (sqrt(state.layers{l}.v{j}) + opts.epsilon),theta);
                    else
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} - lr * state.layers{l}.m{j} ./ (sqrt(state.layers{l}.v{j}) + opts.epsilon);
                    end
                    
                end
            end
        end
end


%%%-------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir,modelName)
%%%-------------------------------------------------------------------------
list = dir(fullfile(modelDir, [modelName,'-epoch-*.mat'])) ;
tokens = regexp({list.name}, [modelName,'-epoch-([\d]+).mat'], 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

%%%-------------------------------------------------------------------------
function A = gradientClipping(A, theta)
%%%-------------------------------------------------------------------------
A(A>theta)  = theta;
A(A<-theta) = -theta;

%%%-------------------------------------------------------------------------
function fn = getBatch
%%%-------------------------------------------------------------------------
fn = @(x,y) getSimpleNNBatch(x,y);

%%%-------------------------------------------------------------------------
function [inputs,inputsf,labels] = getSimpleNNBatch(imdb, batch)
%%%-------------------------------------------------------------------------
inputs = imdb.inputs(:,:,:,batch);
labels = imdb.labels(:,:,:,batch);
inputsf = imdb.inputsf(:,:,batch);



