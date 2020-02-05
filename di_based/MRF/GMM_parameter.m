function[mu,sigma]=GMM_parameter(image,segmentation,class_number)
%高斯混合模型的参数估计
[n,d]=size(image);
mu=zeros(class_number,d);
sigma=zeros(d,d,class_number);
   for i=1:class_number
       Im_i=image(segmentation==i,:);
       [sigma(:,:,i),mu(i,:)]=covmatrix(Im_i);
    end
if isnan(mu(1))
    mu(1)=0;
end
if isnan(mu(2))
    mu(2)=0;
end
if sigma(:,:,1)==0
    sigma(:,:,1)=1;
end
if sigma(:,:,2)==0
    sigma(:,:,2)=1;
end
end


    