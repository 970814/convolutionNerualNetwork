function testHandwrittenNumeralRecognition()

    addpath('../');
% 载入训练好的模型
    load('../report/fileReport 20-Mar-2021 00:44:09.txt');

% 绘制 cost 随 iterations 的变化
    figure(1);
    hold on;
    plot([1:(length(costs))],costs);
    axis([1,length(costs)])
    title('Cost');
    xlabel('iterations');
    ylabel('costs');
    hold off;

% 绘制 train 和 test 集 上 cost 随 epoths 的变化
    figure(2);
    hold on;
    plot([1:(length(USCosts))],USCosts);
    plot([1:(length(testCosts))],testCosts);
    axis([1,length(USCosts)])
    title('Cost');
    legend('train','test');
    xlabel('epoths');
    ylabel('costs');
    hold off;

% 绘制 train 和 test 集 上 accuracy 随 epoths 的变化
    figure(3);
    hold on;
    plot([1:(length(accuracies))],accuracies);
    plot([1:(length(testAccuracies))],testAccuracies);
    axis([1,length(accuracies)])
    title('Accuracy');
    legend('train','test');
    xlabel('epoths');
    ylabel('accuracy');
    hold off;
% 绘制 train 和 test 集 上 耗时 随 epoths 的变化
    figure(4);
    hold on;
    plot([1:(length(perEpothCostTms))],perEpothCostTms);
    axis([1,length(perEpothCostTms)])
    title('CostTime');
    xlabel('epoth(th)');
    ylabel('cost-time');
    hold off;


    betterWB = betterWBs(:,end-1);




%   载入训练集和测试集
    load('../fixDataSets/handwrittenDigit/data')
%    对数据进行归一化
    x=x/255;
    testx=testx/255;
    m=length(Y);
    disp(sprintf('训练集样本大小%d',m))
    testmM=length(testY);
    disp(sprintf('测试集样本大小%d',testmM))

    [bw,bb] = unboxingParameters(betterWB,L,layerTypes,layerNeruals);
%        计算每个epoth后训练集准确率
    accuracy = calClassifyAccuracy(x,Y,bw,bb,L,layerTypes,layerNeruals,ps);
%        计算每个epoth后测试集准确率
    testAccuracy = calClassifyAccuracy(testx,testY,bw,bb,L,layerTypes,layerNeruals,ps);
    fprintf('训练集准确率: %4.6f%%, 测试集准确率: %4.6f%%\n',accuracy*100,testAccuracy*100);
end;