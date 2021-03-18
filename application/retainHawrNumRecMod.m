function retainHawrNumRecMod()
    close all;
    clear all;
    addpath('../');

%    % -1 代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表softmax输出层
%    layerTypes   = [-1,            0,           1              0             1         2            3]
%    % 网络每层神经元的规模
%    layerNeruals = [28,28,1;     24,24,20;     12,12,20;      8,8,40;     4,4,40;    100,1,1;     10,1,1]
%    %池化区域大小
%    ps{3}=[2,2];
%    ps{5}=[2,2];
%    %网络层数
%    L = length(layerTypes);

%   载入训练集和测试集
    load('../fixDataSets/handwrittenDigit/data')
    m=length(Y);
    testmM=length(testY);

%    m = 4;
%    x=x(:,:,1,1:m);
%    y=y(:,1:m);
%    Y=Y(1:m);
%    testmM=10;
%    testx=testx(:,:,1,1:testmM);
%    testy=testy(:,1:testmM);
%    testY=testY(1:testmM);

    disp(sprintf('训练集样本大小%d',m))
    disp(sprintf('测试集样本大小%d',testmM))


% 载入训练好的模型
    load('../report/fileReport 18-Mar-2021 15:05:12.txt');
    wb = betterWBs(:,end);

%    训练模型wb参数
    trainHawrNumRecMod(wb,x,y,Y,testx,testy,testY,L,m,layerNeruals,layerTypes,ps);
end;