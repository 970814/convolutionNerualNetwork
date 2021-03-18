function testNnConvolution()
    addpath('../');
    a=[1 2 3;3 4 5;5 6 7;7 8 9];
    w=[1 2 ;3 4];
%   单张单通道图片   与   单个2d卷积核 进行卷积操作
    z = nnConvolution(a,w)



    % 2个通道数
    A(:,:,1)=a;
    A(:,:,2)=a;

% 一个3d卷积核
    W(:,:,1)=w;
    W(:,:,2)=w;
%  单张多通道图片    与   单个3d卷积核 进行卷积操作
    Z=nnConvolution(A,W)

% 考虑有3张图片
    B(:,:,:,1)=A;
    B(:,:,:,2)=A;
    B(:,:,:,3)=A;

%  多张多通道图片    与   单个3d卷积核  进行卷积操作
    Z=nnConvolution(B,W)

% 给每个样本添加2 个特征
    ZZ (:,:,1,:)=Z;
    ZZ (:,:,2,:)=Z;
    ZZ
    size(ZZ)

end;



