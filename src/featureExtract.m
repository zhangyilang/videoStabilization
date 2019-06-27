function [feature_i,feature_p] = featureExtract(video)
%FEATUREXTRACT: feature extractor for motion estimation
% Input:
%   video: a cell whose every element is a frame (3-d matrix of [width, height, channel]).
% Outpur:
%   feature: a cell whose every element is a feature vector or matrix.

frames = cell(length(video),1);
for i = 1:length(video)
    frames{i} = rgb2gray(video{i});
end
frameTotal = length(frames);
feature_i = cell(frameTotal - 1, 1);
feature_p = cell(frameTotal - 1, 1);
% points1 = fDetect(frame1);
% points2 = fDetect(frame2);
% [index1,indey1]=find(points1);
% index1=[index1 indey1];
% points1=cornerPoints(index1);
% 
% [index2,indey2]=find(points2);
% index2=[index2 indey2];
% points2=cornerPoints(index2);
for frameIndex = 1 : frameTotal - 1
    frame1 = frames{frameIndex};
    frame2 = frames{frameIndex + 1};
    points1 = cornerDetector('Shi_Tomasi',frame1);
    points2 = cornerDetector('Shi_Tomasi',frame2);

    [features1,valid_points1] = extractFeatures(frame1, points1);
    [features2,valid_points2] = extractFeatures(frame2, points2);

    indexPairs = matchFeatures(features1, features2);

    feature_i{frameIndex} = valid_points1(indexPairs(:,1),:);
    feature_p{frameIndex} = valid_points2(indexPairs(:,2),:);

%     figure;
%     showMatchedFeatures(frame1,frame2,feature_i{frameIndex},feature_p{frameIndex});
end
end

