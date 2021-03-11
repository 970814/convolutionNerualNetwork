function forwardPropagation(X,L,w,b,layerTypes,layerNeruals,ps)


    size(X)
    a=X;
    for l=2:L,
        l
        if layerTypes(l) == 0,
    %        如果是卷积层

    %        C为当前层卷积核的个数
            C = size(w{l},4);
            z=[];
            for c=1:C,
    %            多张多通道图片 与 3d卷积核 卷积操作
                z(:,:,c,:) = nnConvolution(a , w{l}(:,:,:,c)) .+ b{l}(c);

            end;
            %            a = nonlinearActivateFunction(z);
            a=z;
            size(a)

        elseif layerTypes(l) == 1,
    %        如果是池化层
            r = layerNeruals(l,1);
            c = layerNeruals(l,2);
            z=[];
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

        elseif layerTypes(l) == 3,
    %        如果是softmax输出层

        else
            disp(sprintf('未定义的网络层类型 %d',layerTypes(l)))
            return;
        end;
    end;

end;