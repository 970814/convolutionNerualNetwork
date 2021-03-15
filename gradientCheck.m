function err = gradientCheck(gw,gb,x,y,w,b,L,layerTypes,layerNeruals,ps)

%   梯度检测
    gradientCheck = false;
    err=false;
    if gradientCheck,

        m=size(x,4);
        disp(sprintf('本次检测的样本数量为%d',m));
        epsilon = 0.0001;
        maxErrDiff = 0.0001;
        wc = 0;
        bc = 0;
        for l = 2:L,
%            l


            if layerTypes(l) == 3 || layerTypes(l) == 2 || layerTypes(l) == 0,
                %            检测biases梯度是否正确
                            N = length(b{l});
                            disp('开始检测 biases');
                            for n=1:N,
                                 b{l}(n)=b{l}(n) + epsilon;
                                 costB = forwardPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps);
                                 b{l}(n)=b{l}(n) - 2 * epsilon;
                                 costA = forwardPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps);
                                 %                    使用双侧差分求偏导数
                                 pdb = (costB-costA)/(2*epsilon);
                                 if abs(pdb - gb{l}(n)) > epsilon,

                                     disp(sprintf('偏导数b检测错误在第%d层的第%d个, 期望 %f, 实际是 %f, %f,%f 差距是 %f',l,n,pdb,gb{l}(n),gb{l}(n)/m,gb{l}(n)/m-pdb,abs(pdb - gb{l}(n))));
                %            当检测到第一个错误时，将停止检测
                                     err=true;
%                                     break;
                                 else
                                     bc = bc+1;
                                     disp(sprintf('偏导数b检测正确在第%d层的第%d个, 期望 %f, 实际是 %f, 差距是 %f',l,n,pdb,gb{l}(n),abs(pdb - gb{l}(n))));
                                 end;
                %                    还原biases值
                                 b{l}(n)=b{l}(n) + epsilon;
                            end;
                %            当检测到第一个错误时，将停止检测
                            if err,
                                break;
                            end;
            end;

            if layerTypes(l) == 3 || layerTypes(l) == 2,

    %           检测 weights梯度是否正确
                [R,C]= size(w{l});
                disp('开始检测 weights')
                for r=1:R,
                    for c=1:C,

                        w{l}(r,c)=w{l}(r,c) + epsilon;
                        costB = forwardPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps);
                        w{l}(r,c)=w{l}(r,c) - 2 * epsilon;
                        costA = forwardPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps);
    %                    使用双侧差分求偏导数
                        pdw = (costB-costA)/(2*epsilon);
                        if abs(pdw - gw{l}(r,c)) > maxErrDiff,
                            disp(sprintf('偏导数检测错误在第%d层的第%d行第%d列, 期望 %f, 实际是 %f, 差距是 %f',l,r,c,pdw,gw{l}(r,c),abs(pdw - gw{l}(r,c))));
    %              当检测到第一个错误时，将停止检测
                            err=true;
                            break;
                        else
                            wc = wc+1;
                            disp(sprintf('偏导数检测正确在第%d层的第%d行第%d列, 期望 %f, 实际是 %f, 差距是 %f',l,r,c,pdw,gw{l}(r,c),abs(pdw - gw{l}(r,c))));
                        end;
    %                    还原权重值
                        w{l}(r,c)=w{l}(r,c) +  epsilon;

                    end;
    %              当检测到第一个错误时，将停止检测
                    if err,
                        break;
                    end;
                end;
    %            当检测到第一个错误时，将停止检测
                if err,
                    break;
                end;

            elseif layerTypes(l) == 1,
%            池化层无参数忽略
                 disp(sprintf('第%d层，池化层无参数则忽略',l))
            elseif layerTypes(l) == 0,
%                卷积层的梯度检测
                disp('开始检测卷积层 weights')
%                得到卷积核的四维
                [H,W,C,nC]=size(w{l});
                for s1 = 1:nC,
                    for s2 = 1:C,
                        for s3 = 1:W,
                            for s4 = 1:H,
                                w{l}(s4,s3,s2,s1) = w{l}(s4,s3,s2,s1) + epsilon;
                                costB = forwardPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps);
                                w{l}(s4,s3,s2,s1) = w{l}(s4,s3,s2,s1) - 2 * epsilon;
                                costA = forwardPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps);
            %                    使用双侧差分求偏导数
                                pdw = (costB-costA)/(2*epsilon);
                                if abs(pdw - gw{l}(s4,s3,s2,s1)) > maxErrDiff,
                                    disp(sprintf('卷积层w偏导数检测错误在第%d层的第%d个卷积核的第%d片的第%d行第%d列, 期望 %f, 实际是 %f, 差距是 %f',l,s1,s2,s3,s4,pdw,gw{l}(s4,s3,s2,s1),abs(pdw - gw{l}(s4,s3,s2,s1))));
            %              当检测到第一个错误时，将停止检测
                                    err=true;
%                                    disp('检测到梯度计算错误');
%                                    return;
                                else
                                    wc = wc+1;
                                    disp(sprintf('卷积层w偏导数检测正确在第%d层的第%d个卷积核的第%d片的第%d行第%d列, 期望 %f, 实际是 %f, 差距是 %f',l,s1,s2,s3,s4,pdw,gw{l}(s4,s3,s2,s1),abs(pdw - gw{l}(s4,s3,s2,s1))));
                                end;
            %                    还原权重值
                                w{l}(s4,s3,s2,s1)=w{l}(s4,s3,s2,s1) +  epsilon;

                            end;
                        end;
                    end;
                end;
            else
                disp(sprintf('未定义的网络层类型 %d',layerTypes(l)))
                return;
            end;
        end;
        disp(sprintf('%d 个w的偏导数计算正确,%d 个b的偏导数计算正确',wc,bc));
        if err,
            disp('检测到梯度计算错误');
        else
%            gw
%            gb
            disp('检测到梯度计算完全正确');
        end;



    end;
end;