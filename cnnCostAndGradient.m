function [cost,g] = cnnCostAndGradient(x,y,wb,L,layerTypes,layerNeruals,ps)




    [w,b] = unboxingParameters(wb,L,layerTypes,layerNeruals);

    [cost,gw,gb] = backPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps);





%    gradientCheck(w,b)

end;