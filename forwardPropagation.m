function cost = forwardPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps)
%    x 输入图片数据，为H*W*C*M多维数组, 分别为高、宽、通道数、样本数
%    y 为图片标签，  为T*M 的矩阵，分别为类别个数、样本数
%    L 为网络层数，
%    w、b为学习参数，w{l}代表l层的参数
%    layerTypes 标记了每层的层类型，-1代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表输出层
%    layerNeruals 记录了每层的神经元个数
%    ps 记录每层的池化大小

    t0 = time();
    m=size(x,4);
    a=x;
    for l=2:L,


        if layerTypes(l) == 0,
    %        如果是卷积层

    %        C为当前层卷积核的个数
            C = size(w{l},4);
            z  = zeros(layerNeruals(l,1),layerNeruals(l,2),layerNeruals(l,3),m);

            for c=1:C,
    %            多张多通道图片 与 3d卷积核 卷积操作
                z(:,:,c,:) = nnConvolution(a , w{l}(:,:,:,c)) .+ b{l}(c);

            end;
            a = nonlinearActivateFunction(z);


        elseif layerTypes(l) == 1,
    %        如果是池化层
            r = layerNeruals(l,1);
            c = layerNeruals(l,2);
            z=[];
    %  默认采用最大池化，
            for j=1:c,
                for i=1:r,
                    t = a([((i-1)*ps{l}(1)+1):(i*ps{l}(1))],[((j-1)*ps{l}(2)+1):(j*ps{l}(2))],:,:);

%                    maxV=[];
%                    for s1 =  1:size(t,4),
%                        for s2 =1: size(t,3),
%
%                            V = -inf;
%                            for s3 = 1:ps{l}(2),
%                                for s4= 1:ps{l}(1),
%%                                    更新最大值
%                                    if t(s4,s3,s2,s1) > V,
%                                        V=t(s4,s3,s2,s1);
%                                    end;
%                                end;
%                            end;
%                            maxV(:,:,s2,s1) = [V];
%                        end;
%                    end;
%                     修改成该向量化写法，向前传播算法花费数据从14s降低到0.287s
                    C = max(reshape(t,ps{l}(2)*ps{l}(1),1,size(t,3),size(t,4)));

                    z = [z C];
                end;
            end;
            z = reshape(z,r,c,size(t,3),size(t,4));


            a = z;

        elseif layerTypes(l) == 2,
    %        如果是全连接层
%            得到上一层的四维、高、宽、通道数、样本数量
            [H,W,C,M]=size(a);
%           转换成列向量
            a=reshape(a,H*W*C,1,1,M);
%            加权输入
            z = w{l} * a +b{l};
            z=reshape(z,layerNeruals(l,1),layerNeruals(l,2),layerNeruals(l,3),M);
            a = nonlinearActivateFunction(z);
        elseif layerTypes(l) == 3,
    %        如果是softmax输出层,

            if layerTypes(l-1) ~= 2,
                %    如果上一层不是全连接层，则需要变换成列向量

    %            得到上一层的四维、高、宽、通道数、样本数量
                [H,W,C,M]=size(a);
    %           转换成列向量
                a=reshape(a,H*W*C,1,1,M);
    %            加权输入
                z = w{l} * a +b{l};

            else
%                如果上一层是全连接层
                 z = w{l} * a +b{l};
            end;
            z=reshape(z,layerNeruals(l,1),layerNeruals(l,2),layerNeruals(l,3),M);
%           softmax 映射转换成概率分布
%            z 是typeCount*m的矩阵
% 由于指数可能会输出一个inf，因此需要做归一化
%             z = z ./ max(z);
            t = e.^(z);


%            归一化
            a = t ./ sum(t,1);
        else
            disp(sprintf('未定义的网络层类型 %d',layerTypes(l)));
            return;
        end;

    end;

%    因为最后一层是全连接层，因此一个长度为SL的列向量即可表达所有神经元，如果有m个样本，可用SL*m的矩阵表达

    [SL,s1,s2,m]=size(a);
    if s1~=1 ||s2~=1 ,
        disp('检测到非法结果，最后一层神经元必须是SL*1*1*m的结构');
    end;



    a = reshape(a,SL,m,1,1);


% 因此 y也是SL*m的矩阵
    cost = sum(-log(a(find(y==1))))/m;

     t1 = time();

     disp(sprintf('向前传播 %d',t1-t0));
end;



