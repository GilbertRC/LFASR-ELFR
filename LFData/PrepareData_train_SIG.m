%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate training data from SIGGRAPHAsia16_ViewSynthesis_Kalantari_Trainingset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% output: train_SIG.h5  
% uint8 0-255
% ['LFI']   [w,h,aw,ah,N]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

%% path
savepath = 'train_SIG.h5';
folder = './dataset_train_SIG';

listname = './list/Train_SIG.txt';
f = fopen(listname);
if( f == -1 )
    error('%s does not exist!', listname);
end
C = textscan(f, '%s', 'CommentStyle', '#');
list = C{1};
fclose(f); 

%% params
H = 374;
W = 540;

allah = 14;
allaw = 14;

ah = 8;
aw = 8;

%% initialization
LFI = zeros(H, W, ah, aw, 1, 'uint8');
count = 0;

%% generate data
for k = 1:size(list,1)
    lfname = list{k};
    lf_path = sprintf('%s/%s.png',folder,lfname);
    disp(lf_path);
    
    eslf = im2uint8(imread(lf_path));
    img = zeros(H,W,allah,allaw,'uint8');

    for v = 1 : allah
        for u = 1 : allah            
            sub = eslf(v:allah:end,u:allah:end,:);            
            sub = rgb2ycbcr(sub);           
            img(:,:,v,u) = sub(1:H,1:W,1);           
        end
    end
        
    img = img(:,:,4:11,4:11);
      
    % generate patches
    count = count+1;    
    LFI(:, :, :, :, count) = img;
end  
 
%% generate dat
order = randperm(count);
LFI= permute(LFI(:, :, :, :, order),[2,1,4,3,5]); %[h,w,ah,aw,N] -> [w,h,aw,ah,N]

%% writing to HDF5
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

h5create(savepath, '/LFI', size(LFI), 'Datatype', 'uint8'); % width, height, channels, number 
h5write(savepath, '/LFI', LFI);

h5disp(savepath);
