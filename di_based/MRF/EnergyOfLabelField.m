% function [E]=EnergyOfLabelField(segmentation,width,height)
% n=size(segmentation,1);
% segmentation=reshape(segmentation,[width,height]);
% Nei8=imstack2vectors(NeiX(segmentation));
% E=zeros(n,2);
% % potential=imstack2vectors(NeiX(potential));
% 
% for i=1:2
%     E(:,i)= sum((Nei8~=i),2);
% end
% end
%计算空间能量
function E=EnergyOfLabelField(segmentation,width,height,U)
n=size(segmentation,1);
segmentation=reshape(segmentation,[width, height]);
Nei8=imstack2vectors(NeiX(segmentation));
E=zeros(n,2);
%the membership of the centra pixel
Uc=(U(1,:))';
Un=(U(2,:))';
% pixelnum*2 change to r,c,2
U=reshape(U',[width,height,2]);
Uc28=repmat(Uc,[1,8]);
Un28=repmat(Un,[1,8]);
%memberships of the neighbour pixels
Uc8=imstack2vectors(NeiX(U(:,:,2)));
Un8=imstack2vectors(NeiX(U(:,:,1)));
Entropy=-Uc8.*log2(Uc8)-Un8.*log2(Un8);
Entropy(isnan(Entropy))=0;

for i=1:2
  E(:,i)=sum(((Nei8~=i)-(Nei8==i)).*(1-Entropy),2);
%     E(:,i)=sum(Nei8~=i,2);
end
end