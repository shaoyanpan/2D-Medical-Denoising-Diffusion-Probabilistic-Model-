if iscell(true)==0
    True = true;
    true = [];
    for ind = 1:size(True,1)
        %         true{ind} = squeeze(True(ind,1:ori_size(ind,1,1),1:ori_size(ind,1,2),1:ori_size(ind,1,3)));
        true{ind} = squeeze(True(ind,:,:,:));
    end
end
if iscell(prediction)==0
    Prediction = prediction;
    prediction = [];
    for ind = 1:size(Prediction,1)
        %         prediction{ind} = squeeze(Prediction(ind,1:ori_size(ind,1,1),1:ori_size(ind,1,2),1:ori_size(ind,1,3)));
        prediction{ind} = squeeze(Prediction(ind,:,:,:));
    end
end
image = img;
if iscell(img)==0
    Image = img;
    image = [];
    for ind = 1:size(Image,1)
        %         image{ind} = squeeze(Image(ind,1:ori_size(ind,1,1),1:ori_size(ind,1,2),1:ori_size(ind,1,3)));
        image{ind} = squeeze(Image(ind,:,:,:));
    end
end



data = (niftiinfo('C:\Research\Moeen\data\Abdomen\imagesTs\img_50.nii.gz'));
Image = flip(permute(double(squeeze(int16(image{2}))),[2,3,1]),1);
% Prediction = imresize3(int16(niftiread('C:\Research\Pelvic segmentation\data1\Abdomen_split3\labelsTs\label_30.nii.gz')),...
% 1,'nearest');

% Image = double(squeeze((image{2})));
% Prediction = flip(double(squeeze(int16(prediction{2}))),3);
Prediction = flip(permute(double(squeeze(int16(true{2}))),[2,3,1]),1);
% Image = imresize3(Image,[256,256,128],'trilinear');
% Prediction = squeeze(int16(prediction{2}));
Prediction = imresize3(Prediction,[size(Image,1),size(Image,2),size(Image,3)],'nearest');

% Prediction = imresize3(permute(double(squeeze(int16(prediction{4}))),[2,3,1]),[size(Image,1),size(Image,2),size(Image,3)],'nearest');
% Image = imresize3(double(squeeze(image{2})),[256,384,128]);
% Prediction = Prediction(size(Image,1),size(Image,2),size(Image,3));
% Image = double(permute(squeeze(image{8}),[2,3,1]));
% Prediction = int16(permute(prediction{8},[2,3,1]));
% Image = double(permute(squeeze(image{3}),[2,3,1]));
% Prediction = int16(permute(prediction{3},[2,3,1]));

colors={[0.6350 0.0780 0.1840]*255,[132,186,91], [171,104,81], [0.9290 0.6940 0.1250]*255,[0.3010 0.7450 0.9330]*255,...
    [0.8500 0.3250 0.0980]*255,[0 0.4470 0.7410]*255,[0.4940 0.1840 0.5560]*255,[0 1 1]*255,[1 0 1]*255,...
    [1 1 0]*255,[1 0 0]*255,[0 0 1]*255};

% Private abdomen dataset: sample #4 (62) 1,4,5,6,7,8 (155), 2,3 (300)
% Private abdomen dataset: sample #8 (67) 1,5,6,7 (70), 2,3,4,8 (193)
% Private abdomen dataset: sample #7 (66) 1,4,5,6,7 (157), 2,3,8 (270)
% Public abdomen dataset: sample #3 1,2,3,4,6,7,8,9,10,11,12,13 (209),5 (250)
% Public abdomen dataset: sample #5 1,2,3,4,6,7,8,9,10,11,12,13 (169),5 (250)
% Public abdomen dataset: sample #6 1,2,3,4,6,7,8,9,10,11,12,13 (169),5 (220)
slice_index = 30;
I = Image(:,:,slice_index);
P = Prediction(:,:,slice_index);

imshow(I,[]); hold on;
set(gca,'LooseInset',get(gca,'TightInset'));
% label_list = [2,3];
% label_list = [2,3,8];
% label_list = [1,2,3,4,6,7,8,9,10,11,12,13];
label_list = [1,2,3,4,5,6,7,8];
% label_list = [];
for kk = 1:length(label_list)
    k = label_list(kk);
    BW = zeros(size(I));
    BW(squeeze(P)==k)=1;
    [B,L,N,A] = bwboundaries(BW);
    for contour_index = 1:length(B)
        boundary = B{contour_index};
        cidx = k;
        plot(boundary(:,2), boundary(:,1),...
            'Color',[colors{cidx}]/255,'LineWidth',3);
    end
    
end


% aaa = imresize3(uint16(squeeze(prediction(1,1,:,:,:))),[256*2,384*2,128*4],'nearest');
% dicomwrite(reshape(aaa,[256*2,384*2,1,128*4]), 'image.dcm');