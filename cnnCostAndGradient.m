function [cost,g] = cnnCostAndGradient(x,y,wb,L,layerTypes,layerNeruals,ps)



%    解开wb列向量参数为结构型的矩阵
    [w,b] = unboxingParameters(wb,L,layerTypes,layerNeruals);
%    反向传播算法计算梯度和cost
    [cost,gw,gb] = backPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps);


%   进行梯度检测
%    gradientCheck(gw,gb,x,y,w,b,L,layerTypes,layerNeruals,ps);

%   将整个梯度转换成列向量
    g=[];
    for l=2:L,
         g=[g; gw{l}(:); gb{l}(:)];
    end;

end;