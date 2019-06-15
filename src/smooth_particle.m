function [M_smooth] = smooth_particle(M_cummulative, N)
%SMOOTH : smooth the motion estimate with particle filtering.
%   X0 -> X1 -> X2 -> ... -> Xn
%         |     |            |
%         V     V            V
%         Z1    Z2           Zn
% where Xi = [S, ¦È, Tx, Ty]
% Input:
%   M_cummulative: cummulative motion estimation matrix.
%   n: number of particles for each parameter (4n particles in total)
% Output: 
%   M_smooth: cell of smoothed matrices.

FrameLen = length(M_cummulative) + 1;
trans_std = sqrt([1e-3, 1e-3, 1e-3, 1e-3]);   % variance -> standard deviation
observe_std = sqrt([1e-4, 1e-4, 1e-4, 1e-4]);

% transition matrix F and observation matrix H
F = eye(4);
H = eye(4);

% initialization
M_smooth = cell(1, FrameLen);
M_smooth{1} = eye(3);
particles = zeros(4,N);
particles(1,:) = 1;
prob = ones(4,N);

for i = 1:FrameLen-1
    
    % compute observation Zi
    Mi = M_cummulative{i};
    Si = sqrt(Mi(1,1)^2 + Mi(2,1)^2);
    thetai = asin(Mi(2,1)/Si);
    Txi = Mi(1,3);
    Tyi = Mi(2,3);
    Zi = [Si, thetai, Txi, Tyi];
    
    for k = 1:4
        % transition
        particles(k,:) = particles(k,:) + trans_std(k) * randn(1,N);

        % reweight
        prob(k,:) = normpdf(Zi(k), particles(k,:), observe_std(k));
        
        % resample
        particles(k,:) = resample(particles(k,:),prob(k,:));
        
    end
    
    % estimate xi according to particles
    xi = mean(particles,2);
    
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

