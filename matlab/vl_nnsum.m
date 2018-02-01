function Y = vl_nnsum(x1,x2,dzdy)

% --------------------------------------------------------------------
% pixel-level L2 loss
% --------------------------------------------------------------------

if nargin <= 2 || isempty(dzdy)
%     t = ((X-c).^2)/2;
%     Y = sum(t(:))/size(X,4); % reconstruction error per sample;
    Y = x1 + x2;
else
    Y = dzdy;
end

