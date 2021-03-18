function testNnConvolution2()

    a=[ 1,2,3;4,5,6;7,8,9;]
    w=[3 4 ;4 5]

    conv2(a,rot90(w,2),'valid')

    a2=[ 5,5,7;6,8,7;9,8,1;]
    w2  =[9 3 ;2 1]

     conv2(a2,rot90(w2,2),'valid')


    conv2(a,rot90(w,2),'valid') .+conv2(a2,rot90(w2,2),'valid')




    A(:,:,1)=a;
    A(:,:,2)=a2;

% 注意这里w 的顺序是反的
    A
    W(:,:,1)=w2;
    W(:,:,2)=w;
    W


    convn(A,rot90(W,2),'valid')

    W2(:,:,:,1)=W;
    W2(:,:,:,2)=W;

    W2

end;



