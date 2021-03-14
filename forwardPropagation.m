function cost = forwardPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps)
%    x 输入图片数据，为H*W*C*M多维数组, 分别为高、宽、通道数、样本数
%    y 为图片标签，  为T*M 的矩阵，分别为类别个数、样本数
%    L 为网络层数，
%    w、b为学习参数，w{l}代表l层的参数
%    layerTypes 标记了每层的层类型，-1代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表输出层
%    layerNeruals 记录了每层的神经元个数
%    ps 记录每层的池化大小

%    [H,W,C,M]=size(x)
    m=size(x,4);
    a=x;
    for l=2:L,
        l
        z  = zeros(layerNeruals(l,1),layerNeruals(l,2),layerNeruals(l,3),m);
        if layerTypes(l) == 0,
    %        如果是卷积层

    %        C为当前层卷积核的个数
            C = size(w{l},4)

            for c=1:C,
    %            多张多通道图片 与 3d卷积核 卷积操作
                z(:,:,c,:) = nnConvolution(a , w{l}(:,:,:,c)) .+ b{l}(c);

            end;

            size(z)
            a = nonlinearActivateFunction(z);

%            size(a)

        elseif layerTypes(l) == 1,
    %        如果是池化层
            r = layerNeruals(l,1);
            c = layerNeruals(l,2);
    %  默认采用最大池化，
            for j=1:c,
                for i=1:r,
                    t = a([((i-1)*ps{l}(1)+1):(i*ps{l}(1))],[((j-1)*ps{l}(2)+1):(j*ps{l}(2))],:,:);
%修复bug，当采用1层池化时，max(max( 对于 1,1,:,: 类型的数据将会得到单个值,必须显示使用对何维度求最大值
%                    t
%                    t =  max(max(t));% 池化大小必须大于1,
%                    t
%                    size(t)
%                    max(max(t,1),1)


                    maxV=[];
                    for s1 =  1:size(t,4),
                        for s2 =1: size(t,3),

                            V = -inf;
                            for s3 = 1:ps{l}(2),
                                for s4= 1:ps{l}(1),
%                                    更新最大值
                                    if t(s4,s3,s2,s1) > V,
                                        V=t(s4,s3,s2,s1);
                                    end;
                                end;
                            end;
                            maxV(:,:,s2,s1) = [V];
                        end;
                    end;

                    z = [z maxV];
                end;
            end;
%            disp('size a ')
%            size(a)
%            size(z)
            z = reshape(z,r,c,size(t,3),size(t,4));
            a = z;
%            size(a)
        elseif layerTypes(l) == 2,
    %        如果是全连接层
%            得到上一层的四维、高、宽、通道数、样本数量
            [H,W,C,M]=size(a);
%           转换成列向量
            a=reshape(a,H*W*C,1,1,M);
%            size(a)
%            加权输入
            z = w{l} * a +b{l};
            z=reshape(z,layerNeruals(l,1),layerNeruals(l,2),layerNeruals(l,3),M);
            a = nonlinearActivateFunction(z);
%            size(a)
        elseif layerTypes(l) == 3,
    %        如果是softmax输出层,

            if layerTypes(l-1) ~= 2,
                %    如果上一层不是全连接层，则需要变换成列向量

    %            得到上一层的四维、高、宽、通道数、样本数量
                [H,W,C,M]=size(a)
    %           转换成列向量
                a=reshape(a,H*W*C,1,1,M);
                size(a)
    %            加权输入
                z = w{l} * a +b{l};

            else
%                如果上一层是全连接层
                 z = w{l} * a +b{l};
%                 size(z)
            end;
            z=reshape(z,layerNeruals(l,1),layerNeruals(l,2),layerNeruals(l,3),M);
%           softmax 映射转换成概率分布
%            z 是typeCount*m的矩阵
%            z
% 由于指数可能会输出一个inf，因此需要做归一化
%max(z)
%            z = z - max(z)   该归一化不好，
%             z = z ./ max(z);
            t = e.^(z);


%            归一化
            a = t ./ sum(t,1);
%            size(a)
        else
            disp(sprintf('未定义的网络层类型 %d',layerTypes(l)));
            return;
        end;
    end;
%    a
%size(a)

%    a
%    y

    cost = sum(-log(a(find(y==1))))/m;

end;



