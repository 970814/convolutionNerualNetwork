clear all;
close all;


% -1 代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表softmax输出层
layerTypes   = [-1,        0,          1,            0,            1,             2,          3]
% 网络每层神经元的规模
layerNeruals = [64,64,3;   60,60,10;   30,30,10;   26,26,20;     13,13,20;       200,-1,-1;    5,-1,-1]
%[12288   36000    9000   13520    3380     200       5]


% 共享权重、过滤器、卷积核
w{2} = zeros(5,5,3,10);
b{2} = zeros(10,1);
w{4} = zeros(5,5,10,20);
b{4} = zeros(20,1);

%池化区域大小
ps{3}=[2,2];
ps{5}=[2,2];




% 全连接层的参数
w{6} = zeros(200,3380);
b{6} = zeros(200,1);
w{7} = zeros(5,200);
b{7} = zeros(5,1);
%网络层数
L = length(layerTypes);

%向前传播算法
%载入训练集
[X,y,types] = loadFingerTrainData(1);





forwardPropagation(X,L,w,b,layerTypes,layerNeruals,ps)







%testNnConvolution();









%for i=1:2,
%    for j = 1:3,
%        index = (i-1)*3+j;
%        A(:,:,j,i)=reshape(k(i,:),2,2);
%    end;
%end;
%A
%%t=ones(2,2,3,2);
%


%k = [randperm(4);randperm(4);randperm(4);randperm(4);randperm(4);randperm(4)];for i=1:2,for j = 1:3,     index = (i-1)*3+j;     A(:,:,j,i)=reshape(k(index,:),2,2); end;end;B=reshape(A,12,1,2);W = rand(2,12);
%
%
%W*B

