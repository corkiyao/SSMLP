clear;close all;
%% settings
size_input = 72;
size_label = 144;
scale = 2;
stride = 150;
%% initialization
dataa = zeros(size_input, size_input, 128, 1,1);
label = zeros(size_label, size_label, 128, 1,1);
bicubic = zeros(size_label, size_label, 128, 1,1);
count = 0;

%% generate data

test = load('Chikusei_test.mat');

%% 
tic 

    pavia100 = modcrop(test, scale);
    [hei,wid,l] = size(pavia100);
    for x = 1: stride :hei-size_label+1
        for y = 1 :stride : wid-size_label+1
            
            subim_label = pavia100(x : x+size_label-1, y :y+size_label-1,:); 
            dat= gaussian_down_sample(subim_label,scale);
            subim_bicubic = imresize(dat, scale,'bicubic');
            %figure, imshow(subim_label(:,:,100));
            %figure, imshow(subim_input(:,:,100))
            count=count+1;
            dataa(:, :, :,count) = dat;
            label(:, :, :,count) = subim_label;
			bicubic(:, :, :,count) = subim_bicubic;
        end
    end
%save
save('.\your path\chikusei_test-4.mat','dataa','label','bicubic');
