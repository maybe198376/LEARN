function [inputs,inputsf,labels, set] = patches_generation(Dataimg,Dataimgl,mode,max_numPatches,batchSize,fdata)

inputs  = zeros(size(Dataimg,1), size(Dataimg,2), 1, 1,'single');
labels  = zeros(size(Dataimg,1), size(Dataimg,2), 1, 1,'single');
inputsf  = zeros(size(fdata,1), 1, 1,'single');

for i = 1 : size(Dataimg,3)
        im_input = single(Dataimg(:,:,i));
        im_label = single(Dataimgl(:,:,i));
        im_f = single(fdata(:,i));
        inputs(:, :, 1, i)   = im_input;
        labels(:, :, 1, i) = im_label;
        inputsf(:, 1, i) = im_f;
end

inputs = inputs(:,:,:,1:(size(inputs,4)-mod(size(inputs,4),batchSize)));
labels = labels(:,:,:,1:(size(labels ,4)-mod(size(labels ,4),batchSize)));
inputsf = inputsf(:,:,1:(size(inputsf ,3)-mod(size(inputsf ,3),batchSize)));

order  = randperm(size(inputs,4));
inputs = inputs(:, :, 1, order);
labels = labels(:, :, 1, order);
inputsf = inputsf(:, 1, order);
set    = uint8(ones(1,size(inputs,4)));
if mode == 1
    set = uint8(2*ones(1,size(inputs,4)));
end

disp('-------Original Datasize-------')
disp(size(inputs,4));

subNum = min(size(inputs,4),max_numPatches);
inputs = inputs(:,:,:,1:subNum);
labels = labels(:,:,:,1:subNum);
inputsf = inputsf(:,:,1:subNum);
set    = set(1:subNum);

disp('-------Now Datasize-------')
disp(size(inputs,4));















