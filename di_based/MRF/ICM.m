function segmentation=ICM(image,class_number,potential,maxIter)
%条件迭代模式算法（ICM）是解决MRF（马尔科夫）模型的后验分布函数最大化问题的常用方法�?
[width,height,bands]=size(image);
image=imstack2vectors(image);

[segmentation,c]=kmeans(image,class_number); %kmeans对影像进行初始分�?

clear c;
iter=0;

while(iter<maxIter)
    [mu,sigma]=GMM_parameter(image,segmentation,class_number);
    Ef=EnergyOfFeatureField(image,mu,sigma,class_number);
    E1=EnergyOfLabelField(segmentation,potential,width,height,class_number);
    E=Ef+E1;
    ini_map_pre=segmentation;
    [tm,segmentation]=min(E,[],2);% 后验概率�?��转化为能量最小，即取�?���?
    if(isequal(ini_map_pre,segmentation))
        break;
    end
    iter=iter+1;
end
segmentation=reshape(segmentation,[width height]);
end