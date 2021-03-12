clear all;
close all;


% -1 代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表softmax输出层
layerTypes   = [-1,        0,          1,            0,            1,             2,          3 ]
% 网络每层神经元的规模
layerNeruals = [64,64,3;   60,60,10;   30,30,10;   26,26,20;     13,13,20;       200,-1,-1;    6,-1,-1]
%[12288   36000    9000   13520    3380     200       6]


% 共享权重、过滤器、卷积核
w{2} = rand(5,5,3,10);
b{2} = rand(10,1);
w{4} = rand(5,5,10,20);
b{4} = rand(20,1);

%池化区域大小
ps{3}=[2,2];
ps{5}=[2,2];




% 全连接层的参数
w{6} = rand(200,3380);
b{6} = rand(200,1);
w{7} = rand(6,200);
b{7} = rand(6,1);
%网络层数
L = length(layerTypes);

%向前传播算法
%载入训练集
[X,Y,types] = loadFingerTrainData(10);
m=length(Y)
% 数据归一化
% 将unit8转换成0～1之间的double
x = im2double(X);
% 将y转换成向量形式
Y
y = zeros(length(types),m);
for i =1:m,
    y(Y(i)+1,i)=1;
end;
y

forwardPropagation(x,y,L,w,b,layerTypes,layerNeruals,ps)







%testNnConvolution();










