function forwardPropagation(X,L,w,b,layerTypes,layerNeruals,ps)


    size(X)
    a=X;
    for l=2:L,
        l
        z=[];
        if layerTypes(l) == 0,
    %        如果是卷积层

    %        C为当前层卷积核的个数
            C = size(w{l},4);

            for c=1:C,
    %            多张多通道图片 与 3d卷积核 卷积操作
                z(:,:,c,:) = nnConvolution(a , w{l}(:,:,:,c)) .+ b{l}(c);

            end;
            a = nonlinearActivateFunction(z);

            size(a)

        elseif layerTypes(l) == 1,
    %        如果是池化层
            r = layerNeruals(l,1);
            c = layerNeruals(l,2);

            for j=1:c,
                for i=1:r,
                    t = a([((i-1)*ps{l}(1)+1):(i*ps{l}(1))],[((j-1)*ps{l}(2)+1):(j*ps{l}(2))],:,:);
    %
                    t =  max(max(t));% 池化大小必须大于1

                    z = [z t];
                end;
            end;
            z = reshape(z,r,c,size(t,3),size(t,4));
            a = z;
            size(a)
        elseif layerTypes(l) == 2,
    %        如果是全连接层
%            得到上一层的四维、高、宽、通道数、样本数量
            [H,W,C,M]=size(a);
%           转换成列向量
            a=reshape(a,H*W*C,1,M);
            size(a)
%            加权输入
            z = w{l} * a +b{l};
            a = nonlinearActivateFunction(z);
            size(a)
        elseif layerTypes(l) == 3,
    %        如果是softmax输出层,

            if layerTypes(l-1) ~= 2,
                %    如果上一层不是全连接层，则需要变换成列向量

    %            得到上一层的四维、高、宽、通道数、样本数量
                [H,W,C,M]=size(a);
    %           转换成列向量
                a=reshape(a,H*W*C,1,M);
                size(a)
    %            加权输入
                z = w{l} * a +b{l};

            else
%                如果上一层是全连接层
                 z = w{l} * a +b{l};
                 size(z)
            end;
%           softmax 映射转换成概率分布
%            z 是typeCount*m的矩阵
            z
% 由于指数可能会输出一个inf，因此需要做归一化
%max(z)
%            z = z - max(z)   该归一化不好，
%             z = z ./ max(z)
            t = e.^(z)


%            归一化
            a = t ./ sum(t,1);
            size(a)
        else
            disp(sprintf('未定义的网络层类型 %d',layerTypes(l)))
            return;
        end;
    end;
    a
end;