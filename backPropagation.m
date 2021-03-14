function [cost,gw,gb] = backPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps)
%     反向传播算法，计算w和b的梯度

%    x 输入图片数据，为H*W*C*M多维数组, 分别为高、宽、通道数、样本数
%    y 为图片标签，  为T*M 的矩阵，分别为类别个数、样本数
%    L 为网络层数，
%    w、b为学习参数，w{l}代表l层的参数
%    layerTypes 标记了每层的层类型，-1代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表输出层
%    layerNeruals 记录了每层的神经元个数
%    ps 记录每层的池化大小


disp('start')
% 记录样本个数
    m=size(x,4);
%  先进行向前传播算法
    a{1}=x;
    z={};
%    用于记录 最大池化 前 最大值的位置
    maxPoolingLocation={};
    for l=2:L,
        z{l}  = zeros(layerNeruals(l,1),layerNeruals(l,2),layerNeruals(l,3),m);
        if layerTypes(l) == 0,
    %        如果是卷积层

    %        C为当前层卷积核的个数
            C = size(w{l},4);

            for c=1:C,
    %            多张多通道图片 与 3d卷积核 卷积操作
                z{l}(:,:,c,:) = nnConvolution(a{l-1} , w{l}(:,:,:,c)) .+ b{l}(c);

            end;
            a{l} = nonlinearActivateFunction(z{l});


        elseif layerTypes(l) == 1,
    %        如果是池化层
            r = layerNeruals(l,1);
            c = layerNeruals(l,2);
            zt=[];
    %  默认采用最大池化，
            maxPoolingLocation{l}=[];
            for j=1:c,
                for i=1:r,
                    t = a{l-1}([((i-1)*ps{l}(1)+1):(i*ps{l}(1))],[((j-1)*ps{l}(2)+1):(j*ps{l}(2))],:,:);

                    mpl=[];
                    maxV=[];


                    for s1 =  1:size(t,4),
                        for s2 =1: size(t,3),

                            maxIndexR=-1;
                            maxIndexC=-1;
                            V = -inf;
                            for s3 = 1:ps{l}(2),
                                for s4= 1:ps{l}(1),
%                                    更新最大值 和位置
                                    if t(s4,s3,s2,s1) > V,
                                        V=t(s4,s3,s2,s1);
                                        maxIndexR= s4;
                                        maxIndexC= s3;
                                    end;
                                end;
                            end;
                            mpl(:,:,s2,s1)=[maxIndexR,maxIndexC];
                            maxV(:,:,s2,s1) = [V];
                        end;
                    end;
%                    size(maxV)
%                    size(mpl)


%                    转换乘 1维度的列向量，计算最大值和最大值的位置
%                    t = reshape(t,ps{l}(1)*ps{l}(2),1, size(t,3), size(t,4));
%                    9
%                    t
%
%                    max(t,1)
%                    [maxV,maxIndex]=max(t);


%                    将最大值的位置变换成2维度的,并记录最大池化的原位置
%                    mpl=[mod(maxIndex-1,ps{l}(1))+1,floor((maxIndex-1)/ps{l}(1))+1];



%                    将池化结果合并
                    zt = [zt maxV];
                    maxPoolingLocation{l} =[maxPoolingLocation{l}; mpl];

                end;
            end;
%            disp('maxPoolingLocation')
%            size(maxPoolingLocation{l})


            z{l} = reshape(zt,r,c,size(t,3),size(t,4));
%             size( z{l})
            a{l} = z{l};
        elseif layerTypes(l) == 2,
    %        如果是全连接层
%            得到上一层的四维、高、宽、通道数、样本数量，默认认为上层为池化层或卷积层
%            disp('heeee')
            [H,W,C,M]=size(a{l-1});
%           转换成列向量
            at = reshape(a{l-1},H*W*C,1,1,M);

%            size(at)
%            加权输入
            z{l} = w{l} * at +b{l};
            z{l}=reshape(z{l},layerNeruals(l,1),layerNeruals(l,2),layerNeruals(l,3),M);
%            size(z{l})
            a{l} = nonlinearActivateFunction(z{l});
        elseif layerTypes(l) == 3,
    %        如果是softmax输出层,

            if layerTypes(l-1) ~= 2,
                %    如果上一层不是全连接层，则需要变换成列向量

    %            得到上一层的四维、高、宽、通道数、样本数量
                [H,W,C,M]=size(a{l-1});
    %           转换成列向量
                at=reshape(a{l-1},H*W*C,1,1,M);
%                size(at)
    %            加权输入
                z{l} = w{l} * at +b{l};

            else
%                如果上一层是全连接层
                 z{l} = w{l} * a{l-1} +b{l};
            end;
            z{l}=reshape(z{l},layerNeruals(l,1),layerNeruals(l,2),layerNeruals(l,3),M);
%           softmax 映射转换成概率分布
%            z 是typeCount*m的矩阵
%            z{l}
% 由于指数可能会输出一个inf，因此需要做归一化
%max(z)
%            z{l} = z{l} ./ max(z{l});
            t = e.^(z{l});


%            归一化
            a{l} = t ./ sum(t,1);
        else
            disp(sprintf('未定义的网络层类型 %d',layerTypes(l)))
            return;
        end;
    end;
%    a{L}

%    y
%    m为样本数量
%    m=size(a{L},4);
    cost = sum(-log(a{L}(find(y==1))))/m;

%    for l =1:L,
%        size(z{l})
%        size(a{l})
%    end;


%    a
%    z

%    z

%    进行反向传播算法
%   默认最后一层为softmax层，所以最后一层的误差为
    Delta{L} = a{L}-y;

%a{L}
444
    a{L}
    Delta{L}
    for l=(L-1):-1:2,
        l
%        Delta{l} = zeros(size(z{l}))
        if layerTypes(l+1) == 3 || layerTypes(l+1) == 2,
%            如果当前层的下一层是softxmax或全连接层，
            Delta{l} = (w{l+1}' * Delta{l+1});
            1223
            size(w{l+1})
            size(Delta{l+1})
%           暂不考虑当前隐藏层为softmax层的情况，
%            z{l}保留了层的结构，例如为池化层时，结构是多维数组，全连接层时，为2维矩阵
%           但计算出的Delta{l}默认为2维矩阵结构，因此需要相应的将Delta{l}转换成当前层的结构。
%           因此计算z的4维
            [H,W,C,M]=size(z{l})
            Delta{l} = reshape(Delta{l},H,W,C,M);
            if layerTypes(l) ~= 1,
%                如果当前层不是池化层，则需要乘以激活函数的导数，否则池化层激活函数为f(x)=x,导数为1，不必乘。
                Delta{l} = Delta{l} .* derNonLinActFun(z{l});
            end;

        elseif layerTypes(l+1) == 1,
%            如果当前层的下一层是池化层，
%           因此计算当前层z的4维
            [H,W,C,M]=size(z{l});
            [H2,W2,C2,M2]=size(z{l+1});


            Delta{l} = zeros(H,W,C,M);
%            mpl记录了这一层池化池化时，最大值的位置，是    神经元个数*2*C*M 的维度
            mpl = maxPoolingLocation{l+1};
            nerualsCount = size(mpl);
%            每个元素
            for s1 = 1:M,
%                每个通道
                for s2 =1:C,
%                    进行unsample 操作
%                    disp('start')
                    for w2 = 1:W2,
                        for h2 = 1:H2,
%                           将池化层元素位置还原到上一层卷积层相应元素起始位置上,然后根据最大元素位置定位
%                            (w2-1)*H2+h2
                            s3 = (h2-1)*ps{l+1}(1)+mpl((w2-1)*H2+h2,1,s2,s1);
                            s4 = (w2-1)*ps{l+1}(2)+mpl((w2-1)*H2+h2,2,s2,s1);
%                           修复池化层反向传播的bug，遗漏了a关于z的偏导数项
                            Delta{l}(s3,s4,s2,s1) = Delta{l+1}(h2,w2,s2,s1) .* derNonLinActFun(z{l}(s3,s4,s2,s1));
                        end;
                    end;
%                     disp('end')
                end;
            end;


%            Delta{l}(:,:,2,2)
%            Delta{l+1}(:,:,2,2)
%            a{l}(:,:,2,2)
%            disp('end2')
        elseif layerTypes(l+1) == 0,


            disp('never happen')
%            如果当前层的下一层是卷积层，


%           因此计算当前层z的4维，意味这Delta也是该维度
            [H,W,C,M]=size(z{l});


%           统计有多少个卷积核
            featureMapCount = size(w{l+1},4);

%            disp('here')


%           考虑每一片误差
            for j = 1:C,

%              考虑所有样本
                for k =1:M,
                    Delta{l}(:,:,j,k) = zeros(H,W);
                    for i=1:featureMapCount,
%                此时定义就为数学上的全卷积操作，旋转360度等于没转
%                2维全卷积操作
                        Delta{l}(:,:,j,k) =Delta{l}(:,:,j,k) + (conv2(Delta{l+1}(:,:,i,k),w{l+1}(:,:,j,i),'full') .* derNonLinActFun(z{l}(:,:,j,k)));
                    end;
                end;
            end;
%            size(Delta{l+1})
%            size(Delta{l})
        else
            disp(sprintf('未定义的网络层类型 %d',layerTypes(l)));
            return;
        end;
    end;


%    for l =1:L,
%        size(Delta{l})
%    end;

%    计算梯度
    for l=2:L,
        if layerTypes(l) == 3 || layerTypes(l) == 2,
%            如果当前层是softxmax或全连接层，

            [H,W,C,M]=size(a{l-1});
%           转换成列向量
            at = reshape(a{l-1},H*W*C,1,1,M);

%            l
%            size(at)
%           size(Delta{l})
%at
%Delta{l}
           gw{l}  =zeros(size(Delta{l},1),size(at,1));
           gb{l}  =zeros(size(Delta{l},1),1);
           for i =1:M,
%                Delta{l}(:,:,1,i)
%               at(:,:,1,i)'
               gw{l} = gw{l} + Delta{l}(:,:,1,i) * at(:,:,1,i)';
               gb{l}  = gb{l} + Delta{l}(:,:,1,i);
           end;
           gw{l} = gw{l} ./ m;
           gb{l} = gb{l} ./ m;

% gw{l}
% gb{l}
%            计算相应的梯度
%            gw{l} = Delta{l} * at' ./ m;


%gb{l}
%133

%Delta{l}
%sum(Delta{l},4)./m
%            gb{l} = sum(Delta{l},4) ./ m;
        elseif  layerTypes(l) == 1,
%            池化层忽略
        elseif layerTypes(l) == 0,
%             如果当前层为卷积层
%            获取当前误差的四维
            [H,W,C,M] = size(Delta{l})


%                计算C个偏置项梯度
            for j =1:C,
                gb{l}(j,1) = 0;
                for k=1:M,
                    gb{l}(j,1)= gb{l}(j,1) + sum(sum(Delta{l}(:,:,j,k)));
                end;
                gb{l}(j,1) = gb{l}(j,1) ./ M;
            end;

%           计算卷积核的4维,C2维卷积核的通道数，M2为卷积核的个数
            [H2,W2,C2,M2] = size(w{l})



%                计算w梯度

            for j=1:C2,
                for i = 1:M2,
%                   计算第i个卷积核的第j片梯度,
                    gw{l}(:,:,j,i) = zeros(H2,W2);
%                    计算k个样本梯度的平均值
                    m
                    for k=1:m,
%                        进行二维卷积操作
                        gw{l}(:,:,j,i) =gw{l}(:,:,j,i) + nnConvolution(a{l-1}(:,:,j,k),Delta{l}(:,:,i,k));
%                        gw{l}(:,:,j,i) =gw{l}(:,:,j,i) + conv2(a{l-1}(:,:,j,k),rot90(Delta{l}(:,:,i,k),2),'valid');
%                        gw{l}(:,:,j,i) =conv2(a{l-1}(:,:,j,k),rot90(Delta{l}(:,:,i,k),2),'valid');
%                        AAA=a{l-1}(:,:,j,k)
%                        BBB= rot90(Delta{l}(:,:,i,k),2)
                    end;
                    gw{l}(:,:,j,i) = gw{l}(:,:,j,i) ./ m;
                end;
            end;
        else
            disp(sprintf('未定义的网络层类型 %d',layerTypes(l)));
            return;
        end;
    end;


    AL = a{L}
%    for l =1:L,
%        l
%        size(gw{l})
%        size(gb{l})
%    end;
%gw
end;




%开始检测卷积层 weights
%卷积层w偏导数检测错误在第2层的第1个卷积核的第1片的第1行第1列, 期望 0.027154, 实际是 0.000000, 差距是 0.027154
%检测到梯度计算错误
%卷积层w偏导数检测错误在第2层的第1个卷积核的第2片的第1行第1列, 期望 0.000000, 实际是 0.027154, 差距是 0.027154


