function retainHawrNumRecMod()
    close all;
    clear all;
    addpath('../');

%   载入训练集和测试集
    load('../fixDataSets/handwrittenDigit/data')
    m=length(Y);
    testmM=length(testY);
%    对数据进行归一化
    x=x/255;
    testx=testx/255;
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


% 载入训练好的模型,还存储了相应的网络配置
    load('../report/fileReport 18-Mar-2021 21:38:36.txt');
    wb = betterWBs(:,end);

%    训练模型wb参数
    trainHawrNumRecMod(wb,x,y,Y,testx,testy,testY,L,m,layerNeruals,layerTypes,ps);
end;