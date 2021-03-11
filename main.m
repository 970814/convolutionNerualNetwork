clear all;
close all;


% -1 代表输入层，0代表卷积层，1代表池化层，2代表全连接层，3代表softmax输出层
layerTypes   = [-1,        0,          1,            0,            1,             2,          3]
% 网络每层神经元的规模
layerNeruals = [64,64,3;   60,60,10;   30,30,10;   26,26,20;     13,13,20;       200,-1,-1;         5,-1,-1]
%[12288   36000    9000   13520    3380     200       5]


% 共享权重、过滤器、卷积核
w{2} = zeros(5,5,3,10);
b{2} = zeros(10);
w{4} = zeros(5,5,10,20);
b{4} = zeros(20);

%池化区域大小
ps{3}=[2,2];
ps{5}=[2,2];




% 全连接层的参数
w{6} = zeros(200,3380);
b{6} = zeros(200);
w{7} = zeros(5,200);
b{7} = zeros(5);
%网络层数
L = length(layerTypes);

%向前传播算法
%载入训练集
[X,y,types] = loadTrainData(2);

imshow(X(:,:,:,1))
figure(2)
imshow(X(:,:,:,2))



forwardPropagation(X,L,w,b,layerTypes,layerNeruals,ps)


%return
%a=X;
%for l=2:L,
%
%    if layerTypes(l) == 0,
%%        如果是卷积层
%
%%        C为当前层卷积核的个数
%        C = size(w{l},4);
%        for c=1:C,
%%            多张多通道图片 与 3d卷积核 卷积操作
%            z(:,:,c,:) = nnConvolution(a , w{l}(:,:,:,c)) .+ b{l}(c);
%        end;
%        %            a = nonlinearActivateFunction(z);
%        a=z;
%    elseif layerTypes(l) == 1,
%%        如果是池化层
%        h = layerNeruals(l,1);
%        w = layerNeruals(l,2);
%        r = h/ps{l}(1);
%        c = w/ps{l}(1);
%        for i=1:r,
%            for j=1:c,
%                t = a([(i-1)*ps{l}(1)+1:i*ps{l}(1)],[(j-1)*ps{l}(2)+1:j*ps{l}(2)],:,:);
%                z(i,j,:,:) = max(max(t));% 池化大小必须大于1
%            end;
%        end;
%    elseif layerTypes(l) == 2,
%%        如果是全连接层
%
%    elseif layerTypes(l) == 3,
%%        如果是softmax输出层
%
%    else
%        disp(sprintf('未定义的网络层类型 %d',layerTypes(l)))
%        return;
%    end;
%end;





%testNnConvolution();







