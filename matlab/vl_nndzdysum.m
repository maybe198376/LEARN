function Y = vl_nndzdysum(x,dzdy,dzdy1)

% --------------------------------------------------------------------
% pixel-level L2 loss
% --------------------------------------------------------------------

if nargin <= 2 || isempty(dzdy)
%     t = ((X-c).^2)/2;
%     Y = sum(t(:))/size(X,4); % reconstruction error per sample;
    Y = x;
else
    Y = (dzdy + dzdy1);
end

