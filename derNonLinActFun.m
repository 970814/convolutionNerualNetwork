function d = derNonLinActFun(z)
% 激活函数的导数


%    a = max(0,z); 函数的导数
%    if z >= 0,
%        d = 1;
%    else
%        d = 0;
%    end;

     a = nonlinearActivateFunction(z);
     d = a  .* (1 - a);

end;
