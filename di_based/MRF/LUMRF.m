clc;
clear all;
close all;
% Read the two instant images and get the difference map Rou
addpath('D:\');
%% Data 1
 Data85=enviread('D:\08');
 Data05=enviread('D:\09');
%% Data 2
% Data85=enviread('F:\Image Data\ETMfarmland\2001');
% Data05=enviread('F:\Image Data\ETMfarmland\2002');
tic;
%% Data 3
%Data85=enviread('F:\Image Data\Exp2\3');
%Data05=enviread('F:\Image Data\Exp2\4');
%% 
DiffData=abs(Data05-Data85);
DiffData=sqrt(DiffData(:,:,1).^2+DiffData(:,:,2).^2+DiffData(:,:,3).^2);
imshow(DiffData,[]);
figure;
% Size of the data,r:Xsize,c:Ysize,b:band No.
[r,c,b]=size(DiffData);
%% initial change map by FCM
data=reshape(DiffData,r*c,b); 
% classify as two clusters ,change and unchange
% Center:cluster No.*band No.  U:2*160000
[center,U,obj_fun]=fcm(data,2);
ini_map=ones(1,r*c);
% 1 unchange, 2 change
ini_map(find(U(1,:)<=U(2,:)))=2;
imshow(reshape(ini_map,r,c),[]);
FCMimage=reshape((ini_map-1).*255,r,c);
toc;
%% Markov 
ini_map=ini_map';
iter=0;
class_number=2;
maxIter=50;

while(iter<maxIter)
    [mu,sigma]=GMM_parameter(data,ini_map,2);
    Ef=EnergyOfFeatureField(data,mu,sigma);
    E1=EnergyOfLabelField(ini_map,r,c,U);
    E=Ef+2*E1;
    ini_map_pre=ini_map;
    [tm,ini_map]=min(E,[],2);
    if(isequal(ini_map_pre,ini_map))
        break;
    end
    
    iter=iter+1;
end
figure;
ini_map=reshape(ini_map,[r,c]);
ini_map=(ini_map-1).*255;
imshow(ini_map,[]);
%figure;
toc;
%% Accuracy assessmentF:\Image Data\ETMfarmland\reference change map_field.tif
%  Cmap=im2double(imread('F:\Image Data\Ice400\Coference change map\v-2 change map.tif'))*255;
% Cmap=imread('F:\Image Data\Exp2\Reference map used.png');
% Cmap=im2double(Cmap(:,:,2))*255;

% imshow(Cmap,[]);

%[poe,pfa,pma,ma,fa,tota]=change_and_error(ini_map,Cmap)
