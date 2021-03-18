function [x,y,Y] = imageFix(pathLabels,pathImages)



maxCount=inf;

%参考 http://yann.lecun.com/exdb/mnist/

%读取图片labels
fid=fopen(pathLabels);
magNum_l=fread(fid,1,'int','ieee-be')
%获取labels数量
itemCount_l=fread(fid,1,'int','ieee-be')
%获取labels
Y=fread(fid,min([maxCount itemCount_l]),'uint8','ieee-be');

%读取图片
fid=fopen(pathImages);
magNum_i=fread(fid,1,'int','ieee-be')
%获取图片数量
itemCount_i=fread(fid,1,'int','ieee-be')
%得到图片的高
rows_i=fread(fid,1,'int','ieee-be')
%得到图片的宽
columns_i=fread(fid,1,'int','ieee-be')
%读取图片
images= fread(fid,[rows_i*columns_i min([maxCount itemCount_i])]);

m  = length(Y)
% 使用斜着的图片进行训练
x = reshape(images,rows_i,columns_i,1,m);
y= zeros(10,m);
for i = 1:m,

%    这个转置操作过分花费时间
    x(:,:,1,i) = x(:,:,1,i)';

    y(Y(i)+1,i) = 1;
    fprintf('%d 张完成\n',i);

%        imshow(x(:,:,i))
%        Y(i)
%        pause;
end;



end;




