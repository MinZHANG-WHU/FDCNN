function [ out_img ] = getPCAFusion( T1,T2 )
T12 = double(cat(3,T1,T2));
im_size = size(T1);
N = im_size(1) * im_size(2);
X = reshape(T12,N,[]);
[COEFF,SCORE,latent,tsquare] = princomp(zscore(X));
Y=SCORE(:,1:3);
Y = reshape(Y,im_size);
minA=min(min(min(Y)));maxA=max(max(max(Y)));  
out_img = uint8((Y-minA) * 255 / (maxA-minA));
end

