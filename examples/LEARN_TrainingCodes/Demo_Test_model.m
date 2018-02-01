
clear all;
%Init 
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;
format compact;
addpath(fullfile('data','utilities'));

global ASRmtx  ASRmtx_T
load proMatrix_64.mat;
ASRmtx = systemMatrix;
ASRmtx_T = ASRmtx';
clear systemMatrix;
%%%Init model
load LEARN_MODEL.mat;
net = vl_simplenn_tidy(net);
net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);
net = vl_simplenn_move(net, 'gpu') ;

%Test Img
load testimg.mat;
input = gpuArray(input);

%Reconstruction
res    = vl_simplenn(net,input,inputf,[],[],'conserveMemory',true,'mode','normal');

output = res(end).x;
output = gather(output);
input  = gather(input);

%Show the result
error = (label - output).^2;
PSNR = 10*log10(1/mean(error(:)));
RMSE = RMSE(output,label);
SSIM = ssim_index(round(output*255),round(label*255));
imshow(output);
disp([PSNR,SSIM,RMSE]);
