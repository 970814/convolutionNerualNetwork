function testHandwrittenNumeralRecognition()

    addpath('../');
% 载入训练好的模型
    load('../report/fileReport 18-Mar-2021 15:05:12.txt');

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


    betterWB = betterWBs(:,end);




%   载入训练集和测试集
    load('../fixDataSets/handwrittenDigit/data')
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