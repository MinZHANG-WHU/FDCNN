function [E]=EnergyOfFeatureField(data,mu,sigma)
data = double(data);
n=size(data,1);
E=zeros(n,2);

for i=1:2
    mu_i=mu(i,:);
    sigma_i=sigma(:,:,i);
    diff_i=data-repmat(mu_i,[n,1]);
    E(:,i)=sum(diff_i*inv(sigma_i).*diff_i,2)+log(det(sigma_i));
end

end
