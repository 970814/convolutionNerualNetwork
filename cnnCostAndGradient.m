function [cost,g] = cnnCostAndGradient(x,y,wb,L,layerTypes,layerNeruals,ps)




    [w,b] = unboxingParameters(wb,L,layerTypes,layerNeruals);
    [cost,gw,gb] = backPropagation(x,y,w,b,L,layerTypes,layerNeruals,ps);




    gradientCheck(gw,gb,x,y,w,b,L,layerTypes,layerNeruals,ps);

end;