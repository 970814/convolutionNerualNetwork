clear all;
close all;


% -1 代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表softmax输出层
layerTypes   = [-1,        0,          1 ,               2,          3 ]
% 网络每层神经元的规模
layerNeruals = [4,4,1;   3,3,2;       3,3,2;             9,1,1;    6,1,1]
%[12288   36000    9000   13520    3380     200       6]

%池化区域大小
ps{3}=[1,1];
ps{5}=[2,2];

%网络层数
L = length(layerTypes);



wb=[];
%计算参数总数量
n=0;
for l=2:L,
    if layerTypes(l) == 0,
%        计算卷积层参数数量
        H = layerNeruals(l-1,1) - layerNeruals(l,1)+1;
        W = layerNeruals(l-1,2) - layerNeruals(l,2)+1;
        c1 = layerNeruals(l-1,3);
        c2 = layerNeruals(l,3);
        n =n+ H*W*c1*c2 + c2;
%         使用方差为 1/H*W*c1 的高斯分布生成w，方差为1 的高斯分布生成b
        wb=  [wb; 1/sqrt(H*W*c1) * randn(H*W*c1*c2,1) ;randn(c2,1)];
    elseif layerTypes(l) == 1,
%            池化层忽略
    elseif layerTypes(l) == 2 || layerTypes(l) == 3,
%           计算全连接层参数数量
        I=layerNeruals(l-1,1)*layerNeruals(l-1,2)*layerNeruals(l-1,3);
        O=layerNeruals(l,1);
        n=n+I*O+O;
%         使用方差为 1/I 的高斯分布生成w，方差为1 的高斯分布生成b
        wb = [wb;1/sqrt(I) * randn(I*O,1) ;randn(O,1)];
    else
        disp(sprintf('未定义的网络层类型 %d',layerTypes(l)))
        return;
    end;
end;
%wb= rand(n,1);
size(wb)
n

%% 共享权重、过滤器、卷积核
%w{2} = rand(5,5,3,10);
%b{2} = rand(10,1);
%w{4} = rand(5,5,10,20);
%b{4} = rand(20,1);
%

%
%
%
%
%% 全连接层的参数
%w{6} = rand(200,3380);
%b{6} = rand(200,1);
%w{7} = rand(6,200);
%b{7} = rand(6,1);


%向前传播算法
%载入训练集
[X,Y,types] = loadFingerTrainData(1);

m=length(Y)
% 数据归一化
% 将unit8转换成0～1之间的double
x = im2double(X);



% 将y转换成向量形式
y = zeros(length(types),m);
for i =1:m,
    y(Y(i)+1,i)=1;
end;


%x =rand(4,4,1,1);


%save wb.txt wb
%save x.txt x
load wb.txt
load x.txt
x
x2 = x;
x(:,:,:,1) = x2;
x(:,:,:,2) = x2;
x


y2 = y;
y(:,:,:,1) = y2;
y(:,:,:,2) = y2;

[w,b] = unboxingParameters(wb,L,layerTypes,layerNeruals);
%相应的增加了x的数量，也需要相应的增加y数量




%w
forwardPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps)
%backPropagation(x,y,L,w,b,layerTypes,layerNeruals,ps)

cnnCostAndGradient(x,y,wb,L,layerTypes,layerNeruals,ps)





%testNnConvolution();










