function [M_smooth] = smooth_particle(M_cummulative, n)
%SMOOTH : smooth the motion estimate with particle filtering.
%   X0 -> X1 -> X2 -> ... -> Xn
%         |     |            |
%         V     V            V
%         Z1    Z2           Zn
% where Xi = [S, sin¦È, Tx, Ty]
% Input:
%   M_cummulative: cummulative motion estimation matrix.
%   n: number of particles
% Output: 
%   M_smooth: cell of smoothed matrices.

FrameLen = length(M_i) + 1;
trans_noise = [1e-3, 1e-3, 1e-3, 1e-3];
observe_noise = [1e-5, 1e-5, 1e-5, 1e-5];

% transition matrix F and observation matrix H and their cov matrices
F = eye(4);
H = eye(4);
sigma_X = diag(trans_noise);
sigma_Z = diag(observe_noise);

% initialization
M_smooth = cell(1, FrameLen);
M_smooth{1} = eye(3);
xi = [1; 0; 0; 0];  % X0
Pi = zero(4);       % init cov matrix of X0


end

