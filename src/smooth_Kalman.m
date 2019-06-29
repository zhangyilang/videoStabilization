function [M_smooth] = smooth_Kalman(M_cummulative)
%SMOOTH : smooth the motion estimate with Kalman filter.
%         i.e. Compute P(Xn|Z1:n)
%   X0 -> X1 -> X2 -> ... -> Xn
%         |     |            |
%         V     V            V
%         Z1    Z2           Zn
% where Xi = [S, ¦È, Tx, Ty]
% Input: 
%   M_i: the estimated transition matrix between two adjacent frames.
%         [ S*cos¦È,-S*sin¦È,Tx;  
%           S*sin¦È, S*cos¦È,Ty;
%             0    ,   0    , 1 ]
%   M_cummulative: cummulative motion estimation matrix.
% Output: 
%   M_smooth: cell of smoothed matrices.

FrameLen = length(M_cummulative) + 1;
trans_noise = [1e-4, 1e-4, 1e-4, 1e-4];     % variance
observe_noise = [1e-1, 1e-1, 1e-1, 1e-1];

% transition matrix F and observation matrix H and their cov matrices
F = eye(4);
H = eye(4);
sigma_X = diag(trans_noise);        % suppose independent
sigma_Z = diag(observe_noise);

% initialization
M_smooth = cell(1, FrameLen);
M_smooth{1} = eye(3);
xi = [1; 0; 0; 0];  % X0
Pi = zeros(4);      % init cov matrix of X0

% Kalman filtering
for i = 1:FrameLen-1
    % compute observation Zi
    Mi = M_cummulative{i};
    Si = sqrt(Mi(1,1)^2 + Mi(2,1)^2);
    thetai = asin(Mi(2,1)/Si);
    Txi = Mi(1,3);
    Tyi = Mi(2,3);
    Zi = [Si, thetai, Txi, Tyi];
    
    % prediction
    xi_pred = F * xi;
    Pi_pred = F * Pi * F'+ sigma_X;
    
    % correction based on observation
    Kg = Pi * H' * (H * Pi * H' + sigma_Z)^-1;  % Kalman gain
    xi = xi_pred + Kg * (Zi' - H * xi_pred);
    Pi = (eye(4) - Kg * H) * Pi_pred;
    
    % compute M_smooth
    Si = xi(1);
    thetai = xi(2);
    sini = sin(thetai);
    cosi = cos(thetai);
    Txi = xi(3);
    Tyi = xi(4);
    M_smooth{i+1} = [Si*cosi, -Si*sini, Txi; ...
                     Si*sini, Si*cosi, Tyi;...
                       0,       0,      1];
end

end
