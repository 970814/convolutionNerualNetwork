function wb = initializeParameters(L,layerTypes,layerNeruals)
%初始化参数成一个长列向量

    wb=[];
    %计算参数总数量
    n=0;
    for l=2:L,
        if layerTypes(l) == 0,
    %        计算卷积层参数数量
            H = layerNeruals(l-1,1) - layerNeruals(l,1)+1;
            W = layerNeruals(l-1,2) - layerNeruals(l,2)+1;
            c1 = layerNeruals(l-1,3);
            c2 = layerNeruals(l,3);
            n =n+ H*W*c1*c2 + c2;
    %         使用方差为 1/H*W*c1 的高斯分布生成w，方差为1 的高斯分布生成b
            wb=  [wb; 1/sqrt(H*W*c1) * randn(H*W*c1*c2,1) ;randn(c2,1)];
        elseif layerTypes(l) == 1,
    %            池化层忽略
        elseif layerTypes(l) == 2 || layerTypes(l) == 3,
    %           计算全连接层参数数量
            I=layerNeruals(l-1,1)*layerNeruals(l-1,2)*layerNeruals(l-1,3);
            O=layerNeruals(l,1);
            n=n+I*O+O;
    %         使用方差为 1/I 的高斯分布生成w，方差为1 的高斯分布生成b
            wb = [wb;1/sqrt(I) * randn(I*O,1) ;randn(O,1)];
        else
            disp(sprintf('未定义的网络层类型 %d',layerTypes(l)))
            return;
        end;
    end;

end;


