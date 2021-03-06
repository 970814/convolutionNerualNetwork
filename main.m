clear all;
close all;


% -1 代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表softmax输出层
layerTypes   = [-1,         0,           1              0             1               2      3 ]
% 网络每层神经元的规模
layerNeruals = [28,28,5;   24,24,10;     12,12,10;      10,10,20;    5,5,20;      5,1,1;   6,1,1]
%[12288   36000    9000   13520    3380     200       6]

%池化区域大小
ps{3}=[1,1];
ps{5}=[2,2];
ps{7}=[3,3];

%网络层数
L = length(layerTypes);


m=1;

x =rand(28,28,5,m);
y=perms([1 0 0 0 0 0])'(:,[1:m]);


wb = initializeParameters(L,layerTypes,layerNeruals);

[w,b] = unboxingParameters(wb,L,layerTypes,layerNeruals);

forwardPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps)

cnnCostAndGradient(x,y,wb,L,layerTypes,layerNeruals,ps)












