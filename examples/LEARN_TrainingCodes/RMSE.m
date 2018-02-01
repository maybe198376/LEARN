function [ f ] = RMSE( h1,h2 )
%RMSE return RMSE(均方根误差)         求两图像的均方根误差
%input must be a imagehandle         输入图像句柄
%image reconstruction evaluation parameter     图像重建评价参数
%    example
%      标准图像 h1
%      比较图像 h2
%      f=RMSE(h1，h2);
%   标准图像与比较图像差异程度，结果越小说明标准图像与比较图像越接近
s=size(size(h1));%判断是灰度图还是RGB
if s(2)==2
    f1=h1;
    f2=h2;
else
    f1=rgb2gray(h1);
    f2=rgb2gray(h2);
end    
G1=double(f1);
G2=double(f2);
[m1,n1]=size(G1);
[m2,n2]=size(G2);
m=min(m1,m2);
n=min(n1,n2);
c=0;
for i=1:m
    for j=1:n
        w=G1(i,j)-G2(i,j);
        c=c+w^2;
    end
end
f=sqrt(c/(m*n));
end

