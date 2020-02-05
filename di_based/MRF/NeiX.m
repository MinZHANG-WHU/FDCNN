function XN=NeiX(X)
%取矩阵的邻域矩阵
[s,t,K]=size(X);
Xu1=zeros(s,t,K);
Xu1(2:s,2:t,:)=X(1:s-1,1:t-1,:);
XN(:,:,1)=Xu1;

Xu=zeros(s,t,K);
Xu(2:s,:,:)=X(1:s-1,:,:);
XN(:,:,2)=Xu;

Xur=zeros(s,t,K);
Xur(2:s,1:t-1,:)=X(1:s-1,2:t,:);
XN(:,:,3)=Xur;

Xr=zeros(s,t,K);
Xr(:,1:t-1,:)=X(:,2:t,:);
XN(:,:,4)=Xr;

Xdr=zeros(s,t,K);
Xdr(1:s-1,1:t-1,:)=X(2:s,2:t,:);
XN(:,:,5)=Xdr;

Xd=zeros(s,t,K);
Xd(1:s-1,:,:)=X(2:s,:,:);
XN(:,:,6)=Xd;

Xd1=zeros(s,t,K);
Xd1(1:s-1,2:t,:)=X(2:s,1:t-1,:);
XN(:,:,7)=Xd1;

X1=zeros(s,t,K);
X1(:,2:t,:)=X(:,1:t-1,:);
XN(:,:,8)=X1;
end