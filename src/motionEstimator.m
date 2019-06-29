function [M_motion] = motionEstimator(feature_i,feature_p)
%UNTITLED: estimate the cummulative motion between each pair of two 
%          adjacent frames.
%          Remember that your task is to estimate the shake in the video,
%          so be careful to exclude the true motion of the video (maybe 
%          this can be achieved by a high pass filter).
% Input:
%   feature: a cell whose every element is a feature vector of
%   corresponding frame.
% Output:
%   M_motion: a cell containing the motion of each frame,
%   the i-th of which is the product of affine transform M_1, ..., M_i.
%   where M_i = [ S*cos¦È,-S*sin¦È,Tx;  
%                 S*sin¦È, S*cos¦È,Ty;
%                   0    ,   0    , 1 ]
M_motion = cell(length(feature_i),1);
for featureNumber = 1 : length(feature_i)
    [M_motion{featureNumber}, ~] = estimateRigidTransformation(feature_i{featureNumber}.Location, feature_p{featureNumber}.Location, 100, 0.2);
end

% M_motion = removeDC(M_motion);

end