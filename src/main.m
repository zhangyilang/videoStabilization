% project of video stabilization for CS:APP
% Group members: 陈幸豪、叶涵诚、王鹏、王鑫、张奕朗
%% Clear
clear
clc
close all

%% Read data
% 生成 VideoReader 实例来读入视频，转为元胞（元胞的每个元素是一帧）
v = VideoReader('../data/01_input.avi');
video = {};
while hasFrame(v)
    video = [video; readFrame(v)];
end

%% Video stabilization
[feature_i,feature_p] = featureExtract(video);
M_motion = motionEstimator(feature_i,feature_p);
M_smooth = smooth_Kalman(M_motion);
% M_smooth = smooth_particle(M_motion, n);

%% motion compensation
% localvideo=video{1:100};
% video_stable = stablize(video, M_motion);
video_stable = shift(video, M_smooth);

%%
writerObj=VideoWriter('../results/result_01.avi');
open(writerObj);
len = length(video_stable);
for i=1:len
    writeVideo(writerObj, video_stable{i}./255);
end
close(writerObj);