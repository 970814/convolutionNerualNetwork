function handwrittenNumeralRecognition()

%
    % -1 代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表softmax输出层
    layerTypes   = [-1,            0,           1              0             1         2            3]
    % 网络每层神经元的规模
    layerNeruals = [28,28,1;     24,24,20;     12,12,20;      8,8,40;     4,4,40;    100,1,1;     10,1,1]
    %池化区域大小
    ps{3}=[2,2];
    ps{5}=[2,2];
    %网络层数
    L = length(layerTypes);

     %载入训练集
    [x,y,Y] = readHandwrittenTrainData(inf);
    m=length(Y);
    disp(sprintf('训练集样本大小%d',m))
%    载入测试集
    [testx,testy,testY] = readHandwrittenTestData(inf);
    testmM=length(testY);
    disp(sprintf('测试集样本大小%d',testmM))

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
    epoths = 10;
    % 迷你训练批次大小,总训练集大小为60000 = 100*600
    miniBatchSize = 100;
    betterWB = wb;
    betterWBs = [];
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

    %   记录每个epoth训练的花费时间
        perEpothCostTm=time()-t0
        perEpothCostTms = [perEpothCostTms   perEpothCostTm];


        [bw,bb] = unboxingParameters(betterWB,L,layerTypes,layerNeruals);
     %    每个epoth 后，计算在整个训练样本上的cost
        USCost =forwardPropagation(x,y,bw,bb,L,layerTypes,layerNeruals,ps)
        USCosts=[USCosts USCost];
    %    每个epoth 后，计算在整个测试样本上的cost
        testCost =forwardPropagation(testx,testy,bw,bb,L,layerTypes,layerNeruals,ps)
        testCosts=[testCosts testCost];
%        记录每次训练后的wb参数
        betterWBs = [betterWBs betterWB];

    end;




    trainCostTime = time() - start


end;