function [out dzdw]= vl_nnCTmapping(x,x1,lmd,dzdy)
global  ASRmtx  ASRmtx_T

if nargin <= 3 || isempty(dzdy)    %forward

    x2 =gather(x);
    clear x;
    x = x2;
   u_temp =  reshape(x,size(x,1)*size(x,2),size(x,4));
   u = double(u_temp);
   f_temp = reshape(x1,size(x1,1),size(x,4));
   f = double( f_temp);      
   ut = u - lmd*ASRmtx_T*(ASRmtx*u - f);
   out_temp = reshape(ut,size(x,1),size(x,2),1,size(x,4));
   out = single(out_temp); 
   out = gpuArray(out); 

else
    x =gather(x);
   u_temp =  reshape(x,size(x,1)*size(x,2),size(x,4));
   u = double(u_temp);
   f_temp = reshape(x1,size(x1,1),size(x,4));
   f = double( f_temp); 
  dzdy = gather(dzdy);
   dzdy_temp =  reshape(dzdy,size(dzdy,1)*size(dzdy,2),size(dzdy,4));
   dzdyd = double(dzdy_temp);
   ut = dzdyd - lmd*ASRmtx_T*(ASRmtx*dzdyd);
   out_temp = reshape(ut,size(dzdy,1),size(dzdy,2),1,size(dzdy,4));
   out = single(out_temp); 
   out = gpuArray(out); 
   lmdt = ASRmtx_T*(ASRmtx*u - f);   
   dzdw = -sum(sum(lmdt.*dzdyd));
   out = gpuArray(out); 
end