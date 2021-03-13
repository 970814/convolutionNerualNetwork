function a = nonlinearActivateFunction(z)

%    a = max(0,z);

     a = 1.0 ./ (1.0 + e.^(-z) );
end;
