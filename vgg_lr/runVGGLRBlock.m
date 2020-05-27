function [ cmm ] = runVGGLRBlock(net,rep_mean, T1_block, T2_block, fusion_img)

dim = 224;

input_t1_m = single(T1_block) - rep_mean;
input_t2_m = single(T2_block) - rep_mean;
net.forward({input_t1_m});

fm_data = net.blob_vec(3);
fm_data = fm_data.get_data();
T1_feauture_map = fm_data;

fm_data = net.blob_vec(6);
fm_data = fm_data.get_data();
T1_feauture_map_112= fm_data;

fm_data = net.blob_vec(10);
fm_data = fm_data.get_data();
T1_feauture_map_56 = fm_data;

net.forward({input_t2_m});
fm_data = net.blob_vec(3);
fm_data = fm_data.get_data();
T2_feauture_map = fm_data;

fm_data = net.blob_vec(6);
fm_data = fm_data.get_data();
T2_feauture_map_112 = fm_data;

fm_data = net.blob_vec(10);
fm_data = fm_data.get_data();
T2_feauture_map_56 = fm_data;
        
DI_224 = abs(T2_feauture_map - T1_feauture_map);
DI_112 = abs(T2_feauture_map_112 - T1_feauture_map_112);
DI_56 =abs(T2_feauture_map_56 - T1_feauture_map_56);

DI_112_224 = imresize(DI_112,2,'nearest');
DI_56_224 = imresize(DI_56,4,'nearest');

%SLIC
[T12_L1,T12_N1] = slicmex(fusion_img,100,60);
[T12_L2,T12_N2] = slicmex(fusion_img,250,60); 
[T12_L3,T12_N3] = slicmex(fusion_img,400,60);

slic_S_1 = getLR(DI_224,64,T12_L1,T12_N1);
slic_S_2 = getLR(DI_112_224,128,T12_L2,T12_N2);
slic_S_3 = getLR(DI_56_224,256,T12_L3,T12_N3);

cm = cat(3,slic_S_1,slic_S_2,slic_S_3);
cm = reshape(cm,[],3);

%multi scale fusion
lamda = double(dim*dim)^(-1/2);%lamda is used to control the weight of the saprsity of S 
[L,S,Y] = singular_value_rpca(cm,lamda);
w = zeros([3,1]);
for i=1:3,
   w(i,1) = norm(S(:,i),1);
end;
w = w/ (max(w(:))-min(w(:))); %normalization
w = exp(-w);
sum_w = sum(w);
w = w ./ sum_w;
cmm = cm * w;
cmm = reshape(cmm,dim,dim);

end

