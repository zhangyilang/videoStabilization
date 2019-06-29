% project of video stabilization for CS:APP
% Group members: 陈幸豪、叶涵诚、王鹏、王鑫、张奕朗
%% Clear
clear
clc
close all

%% Read data
% 生成 VideoReader 实例来读入视频，转为元胞（元胞的每个元素是一帧）
v = VideoReader('../data/03_input.avi');
video = {};
while hasFrame(v)
    video = [video, readFrame(v)];
end

%% Video stabilization
[feature_i,feature_p] = featureExtract(video);
M_motion = motionEstimator(feature_i,feature_p);
M_smooth = smooth_Kalman(M_motion);
% M_smooth = smooth_particle(M_motion, n);

%% 接下来对原来视频中每帧进行处理，将其每个点的坐标扩展后(x,y,1)左乘M_smooth^(-1)，得到(x',y',1)
% localvideo=video{1:100};
video_stable = stablize(video, M_smooth);
% sleep(2)

%%
writerObj=VideoWriter('test_03.avi');
open(writerObj);
figure;
for i=1:length(video_stable)
%     v = imcrop(video_stable{i},[320 180 960 960]);
    imshow(uint8(video_stable{i}(1:360,1:640,:)));
    frame = getframe;
    writeVideo(writerObj,frame);
end
close(writerObj);