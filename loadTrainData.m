function [X,train_set_y,list_classes] = loadTrainData()

    %载入数据
    load('../dataSets/fingers/train_signs.h5');
    %转换成我们想要的格式数据，高*宽*通道数*图片数量
    B(:,:,:,[1:20])=train_set_x(:,:,:,[1:20]);

    A(:,:,1,:) = B(1,:,:,:);
    A(:,:,2,:) = B(2,:,:,:);
    A(:,:,3,:) = B(3,:,:,:);
    X = rot90(A,-1);

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