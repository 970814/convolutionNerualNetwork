function [w,b] = unboxingParameters(wb,L,layerTypes,layerNeruals)
    index = 0;
    for l = 2:L,
        if layerTypes(l) == 0,
%            如果是卷积层，将参数还原
            H = layerNeruals(l-1,1) - layerNeruals(l,1)+1;
            W = layerNeruals(l-1,2) - layerNeruals(l,2)+1;
            c1 = layerNeruals(l-1,3);
            c2 = layerNeruals(l,3);
            n = H*W*c1*c2;
            w{l}=reshape(wb(index+1:index+H*W*c1*c2),H,W,c1,c2);
            index = index + H*W*c1*c2;
            b{l} = wb(index+1:index+c2);
            index = index + c2;
        elseif layerTypes(l) == 1,
%            池化层忽略
        elseif layerTypes(l) == 2 || layerTypes(l) == 3,
%            全连接层
            I=layerNeruals(l-1,1)*layerNeruals(l-1,2)*layerNeruals(l-1,3);
            O=layerNeruals(l,1);

            w{l} = reshape(wb(index+1:index+I*O),O,I);
            index = index+I*O;
            b{l} = wb(index+1:index+O);
            index = index+O;
        end;
    end;
    index
    size(wb)
    for l=1:L,
        size(w{l})
        size(b{l})
    end;


end