function testCNNGradient()

    addpath('../');
%    测试多通道输入、3个卷积层、3个池化层、2个全连接、1个softmax层的网络 梯度是否正确计算
%    这将对 4948 个w的偏导数验证,79 个b的偏导数验证

    % -1 代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表softmax输出层
    layerTypes   = [-1,         0,           1                 0          1             0          1           2         3 ]
    % 网络每层神经元的规模
    layerNeruals = [44,44,5;   40,40,10;     20,20,10;     18,18,20;      9,9,20;       9,9,40;   3,3,40;    3,1,1;     6,1,1]
    %池化区域大小
    ps{3}=[2,2];
    ps{5}=[2,2];
    ps{7}=[3,3];
    %网络层数
    L = length(layerTypes)
%    初始化网络参数
    wb = initializeParameters(L,layerTypes,layerNeruals);

%   测试样本数量
    m=2;
%    随机生成多通道样本
    x =rand(44,44,5,m);
%    随机的标签值
    y=perms([1 0 0 0 0 0])'(:,[1:m]);
%   将长列向量参数解开层对应的每层结构型矩阵
    [w,b] = unboxingParameters(wb,L,layerTypes,layerNeruals);
%    前向传播测试
    forwardPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps)

    h = predict(x,w,b,L,layerTypes,layerNeruals,ps)
%    反向传播计算梯度，并进行梯度检测
    cnnCostAndGradient(x,y,wb,L,layerTypes,layerNeruals,ps)

end;


