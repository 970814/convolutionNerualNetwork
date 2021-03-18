function fixTestTrainImages()
    [x,y,Y] = imageFix('../../dataSets/handwrittenDigit/train-labels.idx1-ubyte','../../dataSets/handwrittenDigit/train-images.idx3-ubyte');
    [testx,testy,testY] = imageFix('../../dataSets/handwrittenDigit/t10k-labels.idx1-ubyte','../../dataSets/handwrittenDigit/t10k-images.idx3-ubyte');

    sf = sprintf('../fixDataSets/handwrittenDigit/data');
    save("-binary",sf,"x","y","Y","testx","testy","testY");

end;

