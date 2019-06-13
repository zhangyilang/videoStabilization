%% Clear
clear
clc
close all

%% Read data
% 生成 VideoReader 实例来读入视频，转为元胞（元胞的每个元素是一帧）

%% Video stabilization
features = featureExtract(video);
M_cummulative = motionEstimator(features);
M_smooth = smooth(M_cummulative);

