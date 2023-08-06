%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate test data from lytro dataset (30scenes/Occlusions/Reflective)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% uint8 0-255
% ['LFI_ycbcr']   [3,w,h,aw,ah,N]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;


%% path
dataset = 'reflective';
folder = './dataset_test_reflective'; 
savepath = sprintf('test_%s.h5',dataset);

listname = sprintf('./list/Test_%s.txt',dataset);
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
LFI_ycbcr = zeros(H, W, 3, ah, aw, 1, 'uint8');
count = 0;

%% generate data
for k = 1:length(list)
    lfname = list{k};
    lf_path = sprintf('%s/%s.png',folder,lfname);
    disp(lf_path)
    
    eslf = im2uint8(imread(lf_path));  
    img_ycbcr = zeros(H,W,3,allah,allaw,'uint8');    

    for v = 1 : allah
        for u = 1 : allah            
            sub = eslf(v:allah:end,u:allah:end,:);  
            sub = rgb2ycbcr(sub);           
            img_ycbcr(:,:,:,v,u) = sub(1:H,1:W,:);           
        end
    end
        
    img_ycbcr = img_ycbcr(:,:,:,4:11,4:11); %[H,W,3,ah,aw]

    count = count+1;    
    LFI_ycbcr(:, :, :, :, :, count) = img_ycbcr;
      
end  
%% generate dat
LFI_ycbcr = permute(LFI_ycbcr,[3,2,1,5,4,6]);

%% save data
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

h5create(savepath,'/LFI_ycbcr',size(LFI_ycbcr),'Datatype','uint8');
h5write(savepath, '/LFI_ycbcr', LFI_ycbcr);

h5disp(savepath);