clear all;
close all;
A = rand(2,2,2,2)


B = reshape(A,4,1,2,2)

[C D]= max(B)

Delta = zeros(4,1,2,2)




dx = [0:2*2-1]'
dx = dx*4
%Delta(dx+D ) =1

D = reshape(D,4,1);


Delta(dx+D)= [1 2 3 4]'

%A()
%1*1的池化不行
