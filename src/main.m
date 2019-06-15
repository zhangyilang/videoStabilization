%% Clear
clear
clc
close all

%% Read data
% 生成 VideoReader 实例来读入视频，转为元胞（元胞的每个元素是一帧）
v = VideoReader('../data/01_input.avi');
video = {};
while hasFrame(v)
    video = [video, readFrame(v)];
end

%% Video stabilization
features = featureExtract(video);
M_cummulative = motionEstimator(features);
M_smooth = smooth_Kalman(M_cummulative);
% M_smooth = smooth_particle(M_cummulative);

% 接下来对原来视频中每帧进行处理，将其每个点的坐标扩展后(x,y,1)左乘M_smooth^(-1)，得到(x',y',1)

