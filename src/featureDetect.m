clc;
clear;

% Load image
img1 = imread('yhc.jpg');
img2 = imread('±£ÑÐyhc.jpg');

% Convert image to grayscale
img1 = rgb2gray(img1);
% img1 = double(img1);

img2 = rgb2gray(img2);
% img2 = double(img2);

% Detect features
% out1 = fDetect(img1);
% out2 = fDetect(img2);

% Display result
% figure,
% subplot(1,2,1)
% imshow(imread('3.jpg'));
% subplot(1,2,2)
% imshow(out1);
% 
% figure,
% subplot(1,2,1)
% imshow(imread('4.jpg'));
% subplot(1,2,2)
% imshow(out2);

video = {img1,img2};
[feature_i,feature_p] = featureExtract(video);

[transformation, inliers] = estimateRigidTransformation(feature_i{1}.Location, feature_p{1}.Location, 100, 0.1);