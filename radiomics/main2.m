warning off;
clear

data_path = 'E:\zcq\ThyroidUS\data1012\norm\0\1';
mask_path = 'E:\zcq\ThyroidUS\data1012\mask\0\1';
%feat_path = 'E:\zcq\ThyroidUS\data1012\features\0\1';
feat_path = 'E:\zcq\ThyroidUS\data1012\features\0\try';
matfiles = dir(data_path);
file_no = size(matfiles, 1);

addpath(genpath('radiomics-master'));

for i =3:file_no

    filename = matfiles(i).name;
    [path, name] = fileparts(filename);

    if exist([feat_path, '/', num2str(name,'%03d'),'.mat'], 'file')
        continue
    end

    Img = imread([data_path, '\', filename]); %

    BW3 = imread([mask_path, '\', filename]);
    % BW3 = BW3_raw(125: 558, 540: 974, :);
    % BW3 = BW3 > 

    [FeaturesBase]=twoside_GetTextureFeatures(Img, BW3);
    
    %% ?1?7?1?7?0?4?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7
    [ImgA1,ImgH1,ImgV1,ImgD1]=dwt2(Img,'coif1'); %?1?7?1?7?1?7?1?7???1?7?1?7?1?7??,?1?7?1?7?1?7?1?7?0?3?1?7?1?7?1?7?1?7?1?7?1?7
    BW3_1=imresize(BW3,size(ImgA1));
 
    TextureHistFeaturesA_1=twoside_GetTextureFeatures(ImgA1,BW3_1);  % ?1?7?1?7???1?7?1?7?1?7?1?7?1?7?1?7?0?5?1?7?1?7?1?7?1?6???1?7?1?7?1?7?1?7?1?7?1?7?1?7
    TextureHistFeaturesH_1=twoside_GetTextureFeatures(ImgH1,BW3_1);
    TextureHistFeaturesV_1=twoside_GetTextureFeatures(ImgV1,BW3_1);    
    TextureHistFeaturesD_1=twoside_GetTextureFeatures(ImgD1,BW3_1);

    FeatureAll=[FeaturesBase TextureHistFeaturesA_1 TextureHistFeaturesH_1 TextureHistFeaturesV_1 TextureHistFeaturesD_1   ];
    save([feat_path, '/', num2str(name,'%03d'),'.mat'],'FeatureAll','FeaturesBase','TextureHistFeaturesA_1','TextureHistFeaturesH_1','TextureHistFeaturesV_1','TextureHistFeaturesD_1','BW3','Img');
%     Feature_TI_System_elastic(i-2,:)=[FeaturesBase TextureHistFeaturesA_1(:,1:end) TextureHistFeaturesH_1(:,1:end) TextureHistFeaturesV_1(:,1:end) TextureHistFeaturesD_1(:,1:end)];
end
