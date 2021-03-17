function [accuracy]=calClassifyAccuracy(x,Y,w,b,L,layerTypes,layerNeruals,ps)
%计算分类准确率

h = predict(x,w,b,L,layerTypes,layerNeruals,ps);
m = size(x,4);
pres = [];
% 将h转换成数字型
for i =1:m,
    [v,index]=max(h(:,i));
    pres = [pres;index-1];
end;
accuracy = length(find(pres == Y))*1.0 / m;



