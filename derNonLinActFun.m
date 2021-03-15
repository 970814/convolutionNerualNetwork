function d = derNonLinActFun(z)
% 激活函数的导数

    %    a = max(0,z); 函数的导数
    [H,W,C,M]=size(z);
    d = zeros(H*W*C*M,1);
    d(find(z(:)>0),1) = 1;

    d= reshape(d,H,W,C,M);


%    if z >= 0,
%        d = 1;
%    else
%        d = 0;
%    end;

%     a = nonlinearActivateFunction(z);
%     d = a  .* (1 - a);

end;
