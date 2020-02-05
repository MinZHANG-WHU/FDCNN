clear all;
clc;
close all;

addpath 'VGG_LR'
caffe.set_mode_gpu();

net=caffe.Net('vgg16-fcn-rs-deploy.prototxt','VGG16_AID.caffemodel','test');

alphas = [1.35,1.2,1.6,1.3];

tic;
method = 'VGG_LR';

[T1,T2,GT,Sensor,alpha] = getDataset(dataset_index,0);

dim = 224;
im_size = size(T1);
new_h = (im_size(1) / dim + 0.5);
new_w = (im_size(2) / dim + 0.5);
new_h = floor(new_h) * dim;
new_w = floor(new_w) * dim;
T1 = imresize(T1,[new_h,new_w],'nearest');
T2 = imresize(T2,[new_h,new_w],'nearest');

mean_data = [101.43782409500028  104.35751543916642 93.9691908327783];
rep_mean = repmat(mean_data',[1,dim,dim]);
rep_mean = permute(rep_mean,[2,3,1]);
batch_h = new_h / dim;
batch_w = new_w / dim ;

cmm = zeros(new_h,new_w,1);

%PCA fusion
fusion_T12 = getPCAFusion(T1,T2);

bar = waitbar(0,'Please wait...');
for i = 1:batch_h
    for j = 1:batch_w
        progress = (i * batch_h + j - 1 - batch_h)  / (batch_h * batch_w);
        str=['Forward...',num2str(progress * 100),'%'];
        waitbar(progress,bar,str);
        x_offset = 1 + (i - 1) * dim;
        y_offset = 1 + (j - 1) * dim;
        x_end = x_offset + dim - 1;
        y_end = y_offset + dim - 1;
        input_t1 = T1(x_offset:x_end ,y_offset:y_end,:);
        input_t2 = T2(x_offset:x_end ,y_offset:y_end,:);
        input_t12 = fusion_T12(x_offset:x_end ,y_offset:y_end,:);
        [ cmm_block ] = runVGGLRBlock(net,rep_mean, input_t1, input_t2,input_t12);
        cmm(x_offset:x_end,y_offset:y_end,:) = cmm_block;
    end
end
close(bar); 
cmm = imresize(cmm,im_size(1:2),'nearest');

mean_cm = mean2(cmm);
threshold = alpha * mean_cm;
BM = (cmm > threshold);

figure;imshow(cmm,[]);title('CMM');
figure;imshow(BM,[]);title('BM');
figure;imshow(T1,[]);title('T1');
figure;imshow(T2,[]);title('T2');
figure;imshow(GT,[]);title('GT');