function [X,Y,list_classes] = loadFingerTrainData(maxSize)

    %载入数据
    load('../dataSets/fingers/train_signs.h5');
    m = min(maxSize,size(train_set_x,4));
%    至少导入2条数据，不然数据格式会变成列向量
    m =max(2,m);
    %转换成我们想要的格式数据，高*宽*通道数*图片数量
    B(:,:,:,[1:m])=train_set_x(:,:,:,[1:m]);

    A(:,:,1,:) = B(1,:,:,:);
    A(:,:,2,:) = B(2,:,:,:);
    A(:,:,3,:) = B(3,:,:,:);
    X = rot90(A,-1);
%    在这里进行还原，
    m = min(maxSize,m);
    X = X(:,:,:,[1:m]);
    Y = train_set_y([1:m]);

%imshow(X(:,:,:,1));
%   X 为图片数据，是 h*w*c*m 的维度，
%   train_set_y  为图片标签 ，是1*m 的矩阵
%   list_classes  为图片所有可能的类别。


%显示图片
%for i=1:m,
%    imshow(X(:,:,:,i));
%    train_set_y(i)
%    pause;
%end;

end;