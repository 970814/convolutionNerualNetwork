function trainHawrNumRecMod(wb,x,y,Y,testx,testy,testY,L,m,layerNeruals,layerTypes,ps)

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
%      为fminunc高级优化函数设置参数
%    options = optimset('GradObj', 'on', 'MaxIter', 2);
%     训练批次
    epoths = 10
    % 迷你训练批次大小,总训练集大小为60000 = 100*600
    miniBatchSize = 100
    betterWB = wb;
    betterWBs = [];
    accuracies = [];
    testAccuracies = [];
    for e = 1:epoths,
        seg = randperm(m / miniBatchSize);
        t0=time();
        disp(sprintf('Epoth %3i 开始',e));
        for i = 1:length(seg),
            t1=time();
    %       迷你训练批次 x
            trainSet = x(:,:,:,((seg(i)-1)*miniBatchSize+1) :(seg(i)*miniBatchSize));
            trainY = y(:,((seg(i)-1)*miniBatchSize+1) :(seg(i)*miniBatchSize));
            trainLabels = Y(((seg(i)-1)*miniBatchSize+1) :(seg(i)*miniBatchSize));
            costFunction = @(wb) cnnCostAndGradient(trainSet,trainY,wb,L,layerTypes,layerNeruals,ps);
             % 高级优化算法寻找局部最优解
            [betterWB, cost, info] = ...
                fmincg(@(t)costFunction(t), betterWB, options);
%            	fminunc(@(t)costFunction(t), betterWB, options);
            costs = [costs cost];
            ct= time()-t1;
            fprintf('迭代 %4i, Cost: %4.6e, 耗时: %4.6e, Info: %d\n',i, cost, ct, info);
        end;

    %   记录每个epoth训练的花费时间
        perEpothCostTm=time()-t0;
        perEpothCostTms = [perEpothCostTms   perEpothCostTm];


        [bw,bb] = unboxingParameters(betterWB,L,layerTypes,layerNeruals);
     %    每个epoth 后，计算在整个训练样本上的cost
        USCost =forwardPropagation(x,y,bw,bb,L,layerTypes,layerNeruals,ps);
        USCosts=[USCosts USCost];
    %    每个epoth 后，计算在整个测试样本上的cost
        testCost =forwardPropagation(testx,testy,bw,bb,L,layerTypes,layerNeruals,ps);
        testCosts=[testCosts testCost];

%        记录每次训练后的wb参数
        betterWBs = [betterWBs,betterWB];

%        计算每个epoth后训练集准确率
        accuracy=calClassifyAccuracy(x,Y,bw,bb,L,layerTypes,layerNeruals,ps);
%        计算每个epoth后测试集准确率
        testAccuracy=calClassifyAccuracy(testx,testY,bw,bb,L,layerTypes,layerNeruals,ps);

%        记录每个epoth后的准确率
        accuracies = [accuracies accuracy];
        testAccuracies = [testAccuracies testAccuracy];
        fprintf('Epoth %3i 完成, 训练集cost: %4.6e, 测试集cost: %4.6e, 训练集Acc: %3.6f%%, 测试集Acc: %3.6f%%, 耗时: %4.6e\n',...
            e,USCost,testCost,accuracy*100,testAccuracy*100,perEpothCostTm);

    end;




    trainCostTime = time() - start;
    fprintf('训练结束，总耗时: %4.6e\n',trainCostTime);


    fileReport = sprintf('../report/fileReport %s.txt',datestr(now))
    save(fileReport,"L","layerTypes","layerNeruals","ps","betterWBs","accuracies","testAccuracies","perEpothCostTms","costs","USCosts","testCosts");
% 绘制 cost 随 iterations 的变化
    figure(1);
    hold on;
    plot([0:(length(costs)-1)],costs);
    title('Cost');
    xlabel('iterations');
    ylabel('costs');
    hold off;
% 绘制 train 和 test 集 上 cost 随 epoths 的变化
    figure(2);
    hold on;
    plot([0:(length(USCosts)-1)],USCosts);
    plot([0:(length(testCosts)-1)],testCosts);
    title('Cost');
    legend('train','test');
    xlabel('epoths');
    ylabel('costs');
    hold off;

% 绘制 train 和 test 集 上 accuracy 随 epoths 的变化
    figure(3);
    hold on;
    plot([0:(length(accuracies)-1)],accuracies);
    plot([0:(length(testAccuracies)-1)],testAccuracies);
    title('Accuracy');
    legend('train','test');
    xlabel('epoths');
    ylabel('accuracy');
    hold off;
% 绘制 train 和 test 集 上 耗时 随 epoths 的变化
    figure(4);
    hold on;
    plot([0:(length(perEpothCostTms)-1)],perEpothCostTms);
    title('CostTime');
    xlabel('epoth(th)');
    ylabel('cost-time');
    hold off;



end;