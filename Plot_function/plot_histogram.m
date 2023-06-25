
% % names = {'Heart','Left-kidney','Right-kidney','Liver','Left-lung','Right-lung','Spinal-cord','Stomach'};
% names = {'Spleen','Right kidney','Left-kidney','Gallbladder','Esophagus','Liver','Stomach',...
%     'Aorta','Vena cava','Vein','Pancreas','RAG','LAG'};
% % edges=linspace(0,1,20);
% figure
% for ind =1:size(hd_value,2)
%     subplot(5,3,ind)
%     edges=linspace(min(hd_value(:,ind),[],'all'),max(hd_value(:,ind),[],'all'),20);
%     hd_value(10,4) = NaN;
%     histogram(hd_value(:,ind),edges,'Normalization','probability');
%     title(names{ind},'FontSize', 30)
%     xlabel('DSC','FontSize', 15)
%     ylabel('Probability','FontSize', 15)
%     ylim([0,1])
%     set(gca,'FontSize',15)
% end
% all_names = cell(1,12*13);
% names = {'Spleen','Right kidney','Left-kidney','Gallbladder','Esophagus','Liver','Stomach',...
%     'Aorta','Vena cava','Vein','Pancreas','RAG','LAG'};
% % names = {'Heart','Left-kidney','Right-kidney','Liver','Left-lung','Right-lung','Spinal-cord','Stomach'};
% hd_value(10,4) = NaN;
% for ind = 1:size(names,2)
%     all_names((ind-1)*12+1:ind*12) = {names{ind}};
% end
% dice_cao = reshape(hd_value,1,[]);
% figure
% vs = violinplot(dice_cao, all_names,'GroupOrder',names,'ShowMean',true,'ShowMedian',false,'ViolinAlpha',0.2);
% title('Violin plot of HD on BCTV dataset','FontSize', 30)
% xlabel('HD','FontSize', 15)
% ylabel('HD on different organs','FontSize', 15)
% % ylim([0.1,1.05])
% ylim([0 200])
% set(gca,'FontSize',15)

names = {'Spleen','Right kidney','Left-kidney','Gallbladder','Esophagus','Liver','Stomach',...
    'Aorta','Vena cava','Vein','Pancreas','RAG','LAG'};
% names = {'Heart','Left-kidney','Right-kidney','Liver','Left-lung','Right-lung','Spinal-cord','Stomach'};
for organ_index = 1:13
%     label_dir = dir('C:\Research\Pelvic segmentation\data1\Abdomen_all\');
    label_dir = dir('C:\Research\Pelvic segmentation\data1\BCTV\labelsTs\');
    true_volume = [];
    pre_volume = [];
    for ind = 3:length(label_dir)
        scan = niftiread(['C:\Research\Pelvic segmentation\data1\BCTV\labelsTs\',label_dir(ind).name]);
        information = niftiinfo(['C:\Research\Pelvic segmentation\data1\BCTV\labelsTs\',label_dir(ind).name]);
        true_volume = [true_volume, sum(scan==organ_index,'all')*information.PixelDimensions(1)*...
            information.PixelDimensions(2)*information.PixelDimensions(3)/1000];
%         end
    end
    
%     temp_true = true;
%     temp_pre = prediction;
%     temp_true(temp_true~=organ_index)=0;
%     temp_true(temp_true==organ_index)=1;
%     temp_pre(temp_pre~=organ_index)=0;
%     temp_pre(temp_pre==organ_index)=1;
    for ind = 1:size(temp_true,1)
%         cao = [];
%         count = 1;
%         for jnd = 1:size(temp_true(ind,:,:,:),4)
%             if length(unique(temp_true(ind,:,:,jnd)))>1
%                 cao(count) = jnd;
%                 count = count+1;
%             end
%         end
%         pre_sample = temp_pre(ind,:,:,cao);  
%         
        pre_volume = [pre_volume,sum(Prediction(ind,:,:,:)==organ_index,'all')*1.5*1.5*2/1000];
    end
    
    if organ_index == 4
        pre_volume(10) = [];
        true_volume(10) = [];
    end
    x = true_volume;
    y = pre_volume;
%     Fit linear regression line with OLS.
    p = polyfit(x,y,1);
    RegressionLine = polyval(p,x);
    subplot(5,3,organ_index)
    scatter(x,y,55,'o')
    hold on
    plot(x,RegressionLine,'-')
    title(names{organ_index},'FontSize', 20)
    ylabel('Predicted volume (cc)','FontSize', 15)
    xlabel('Ground truth volume (cc)','FontSize', 15)
    if organ_index == 1
        legend('Data','Linear-fit line','FontSize', 10)
    end
    ylim([0, max(max(x),max(y))])
%     ylim([-0, 1.1])
    RMSE = sqrt(mean((y-RegressionLine).^2));
    SS_X = sum((RegressionLine-mean(RegressionLine)).^2);
    SS_Y = sum((y-mean(y)).^2);
    SS_XY = sum((RegressionLine-mean(RegressionLine)).*(y-mean(y)));
    R_squared = SS_XY/sqrt(SS_X*SS_Y);
    text(mean(x),mean(y),['R-square = ',num2str(R_squared)],'FontSize',15)
    set(gca,'FontSize',15)
end