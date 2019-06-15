function [M_smooth] = smooth_Kalman(M_cummulative)
%SMOOTH : smooth the motion estimate with Kalman filter.
%         i.e. Compute P(Xn|Z1:n)
%   X0 -> X1 -> X2 -> ... -> Xn
%         ?     ?            ?
%         Z1    Z2           Zn
% where Xi = [S, sin¦È, Tx, Ty]
% Input: 
%   M_i: the estimated transition matrix between two adjacent frames.
%         [ S*cos¦È,-S*sin¦È,Tx;  
%           S*sin¦È, S*cos¦È,Ty;
%             0    ,   0    , 1 ]
%   M_cummulative: cummulative motion estimation matrix.
% Output: 
%   M_smooth: cell of smoothed matrices.

FrameLen = length(M_i) + 1;
trans_noise = 1e-3;
observe_noise = 1e-5;

% transition matrix F and observation matrix H
% F = zeros(4);
% for i = 1:FrameLen-1
%     M = M_i{i};
%     S = sqrt(M(1,1)^2 + M(1,2)^2);
%     sintheta = M(2,1) / S;
%     Tx = M(1,3);
%     Ty = M(2,3);
%     F = F + diag([S,sintheta,Tx,Ty]);
% end
% F = F / (FrameLen-1);
F = eye(4);
H = eye(4);

% suppose covariance matrix to be identity matrix
sigma_X = trans_noise * eye(4);
sigma_Z = observe_noise * eye(4);

% initialization
M_smooth = cell(1, FrameLen);
M_smooth{1} = eye(3);
xi = [1; 0; 0; 0];  % X0
Pi = zero(4);       % init cov matrix of X0

% Kalman filtering
for i = 1:FrameLen-1
    % prediction
    xi_pred = F * xi;
    Pi_pred = F * Pi * F'+ sigma_X;
    
    % correction based on observation
    Kg = Pi * H' * (H * Pi * H' + sigma_Z)^-1;  % Kalman gain
    xi = xi_pred + Kg * (M_cummulative{i} - H * xi_pred);
    Pi = (eye(4) - Kg * H) * Pi_pred;
    
    % compute M_smooth
    Si = xi(1);
    sini = xi(2);
    cosi = sqrt(1 - sini^2);
    Txi = xi(3);
    Tyi = xi(4);
    M_smooth{i+1} = [Si*cosi, -Si*sini, Txi; ...
                     Si*sini, Si*cosi, Tyi;...
                       0,       0,      1];
end

end
