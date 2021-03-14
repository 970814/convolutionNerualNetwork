function [z] = nnConvolution(a,w)
%    支持单张单通道图片    与     单个2d卷积核    进行卷积操作
%    支持单张多通道图片    与     单个3d卷积核    进行卷积操作
%    支持多张多通道图片    与     单个3d卷积核    进行卷积操作

%    cnn的多维卷积操作

%    a 是一个h*w*c*m 的多维数组，对于图片，h*w为图片大小，c为图片通道数量,m 为图片数量
%    w 是卷积核，是一个x*y*c的多维数组，x*y为卷积核大小（局部接受视野），c为通道数量
%    a和w的通道数量需要匹配
%    产生的结果是一个(h-x+1 * w-y+1)*1*m 的矩阵
%    步幅为1

%    这里依然存在一个问题，对于3d卷积，考虑第三维度a1,...,ai,...,aj,...,ak,...an  与 w1...,wi,...,wj,...,wk,...wn,  的卷积
%    产生的结果是 a1 * wn+ ...+ ai * wk+ ...+ aj * wj + ...+ ak * wi+ ...+an * w1,
%   而我们期待的是a1 * w1+ ...+ ai * wi+ ...+ aj * wj + ...+ ak * wk+... + an * wn,
%    z = convn(a,rot90(w,2),'valid');

    [H,W,C,M] = size(a)
    [H2,W2,C2] = size(w)
%    C2与C必须相等
    if C2 ~= C,
        disp(sprintf('卷积核的通道数必须与输入的通道数相同 %d!=%d',C2,C));
    else
        z = zeros(H-H2+1,W-W2+1,1,M);
        for k =1:M,
%          产生的结果是二维的
%            z(:,:,1,k) = zeros(H-H2+1,W-W2+1);
            for j = 1:C,
                z(:,:,1,k) = z(:,:,1,k) .+ conv2(a(:,:,j,k),rot90(w(:,:,j),2),'valid');
            end;
        end;
        size(z)
        v(:,:,1,:)=z
        v=z
    end;

end;

