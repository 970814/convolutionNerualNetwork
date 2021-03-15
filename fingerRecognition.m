function fingerRecognition()
    clear all;
    close all;

    % -1 代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表softmax输出层
    layerTypes   = [-1,         0,           1              0             1               2           3]
    % 网络每层神经元的规模
    layerNeruals = [64,64,3;   60,60,8;     30,30,8;      26,26,16;     13,13,16;        100,1,1;    6,1,1]
    %池化区域大小
    ps{3}=[1,1];
    ps{5}=[2,2];
    ps{7}=[3,3];
    %网络层数
    L = length(layerTypes);
    %载入训练集
    [X,Y,types] = loadFingerTrainData(1);
    m=length(Y);
    % 数据归一化,将unit8转换成0～1之间的double
    x = im2double(X);
    % 将y转换成向量形式
    y = zeros(length(types),m);
    for i =1:m,
        y(Y(i)+1,i)=1;
    end;

%size(x)
%imshow(x)
%y

    wb = initializeParameters(L,layerTypes,layerNeruals);

    [w,b] = unboxingParameters(wb,L,layerTypes,layerNeruals);

    forwardPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps)

    cnnCostAndGradient(x,y,wb,L,layerTypes,layerNeruals,ps)


end;

