function fingerRecognition()
    clear all;
    close all;

    % -1 代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表softmax输出层
    layerTypes   = [-1,            0,           1              0             1              3]
    % 网络每层神经元的规模
    layerNeruals = [64,64,3;     50,50,6;     25,25,6;      10,10,10;     5,5,10;         6,1,1]
    %池化区域大小
    ps{3}=[2,2];
    ps{5}=[2,2];
    ps{7}=[3,3];
    %网络层数
    L = length(layerTypes);

    %载入训练集
    [x,y,Y,types] = loadFingerTrainData(inf);
    m=length(Y);
    disp(sprintf('训练集样本大小%d,类别%d',m,length(types)))
%    载入测试集
    [testx,testy,testY,test_types] = loadFingerTestData(inf);
    testmM=length(testY);
    disp(sprintf('测试集样本大小%d,类别%d',testmM,length(test_types)))

%展示图片
%size(x)
%imshow(x)
%y

% 初始化参数
    wb = initializeParameters(L,layerTypes,layerNeruals);



    start = time();
%    记录了每次迭代的cost
    costs =[];
%    记录了每个epoth后整个训练样本的cost
    USCosts=[];
%    记录了每个epoth后整个测试样本的cost
    testCosts=[];
%    记录了每个epoth的花费时间
    perEpothCostTms=[];
     % 为fmincg高级优化函数设置参数
    options = optimset('MaxIter', 1);
    % 训练批次
    epoths = 8;
    % 迷你训练批次大小,总训练集大小为1080 = 4*9*3*5*2
    miniBatchSize = 10;
    betterWB = wb;
    for e = 1:epoths,
        seg = randperm(m / miniBatchSize);
        t0=time();
        disp(sprintf('开始第%d个epoth',e));
        for i = 1:length(seg),
            t1=time();
    %         使用全局变量 的迷你训练批次 x
            global trainSet;
            global trainY;
            trainSet = x(:,:,:,((seg(i)-1)*miniBatchSize+1) :(seg(i)*miniBatchSize));
            trainY = y(:,((seg(i)-1)*miniBatchSize+1) :(seg(i)*miniBatchSize));
            trainLabels = Y(((seg(i)-1)*miniBatchSize+1) :(seg(i)*miniBatchSize));
            costFunction = @(wb) cnnCostAndGradient(trainSet,trainY,wb,L,layerTypes,layerNeruals,ps);
             % 高级优化算法寻找局部最优解
            [betterWB, cost, info] = ...
            	fmincg(@(t)costFunction(t), betterWB, options);
            costs = [costs cost];
            ct= time()-t1;
            disp(sprintf('本次迭代花费时间%f',ct));
        end;
        disp(sprintf('第%d个epoth结束',e));
        [bw,bb] = unboxingParameters(betterWB,L,layerTypes,layerNeruals);
     %    每个epoth 后，计算在整个训练样本上的cost
        USCost =forwardPropagation(x,y,bw,bb,L,layerTypes,layerNeruals,ps)
        USCosts=[USCosts USCost];
    %    每个epoth 后，计算在整个测试样本上的cost
        testCost =forwardPropagation(testx,testy,bw,bb,L,layerTypes,layerNeruals,ps)
        testCosts=[testCosts testCost];


%    %    每个epoth 后，计算在整个训练样本上的准确率
%        [errRate,accuracy]=calClassifyErrRate(betterWB,X,labels,nnInfo)
%        errRates= [errRates errRate];
%        accuArr= [accuArr accuracy];
%    %    每个epoth 后，计算在整个测试样本上的准确率
%        [testErrRate,testAccuracy]=calClassifyErrRate(betterWB,testX,testLabels,nnInfo)
%        testErrRates= [testErrRates testErrRate];
%        testAccuArr= [testAccuArr testAccuracy];
    %   记录每个epoth训练的花费时间
        perEpothCostTm=time()-t0
        perEpothCostTms = [perEpothCostTms   perEpothCostTm];
    end;




    trainCostTime = time() - start


end;


