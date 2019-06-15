function [M_smooth] = smooth_particle(M_cummulative)
%SMOOTH : smooth the motion estimate with particle filtering.
%   X0 -> X1 -> X2 -> ... -> Xn
%         ?     ?            ?
%         Z1    Z2           Zn
% where Xi = [S, sin¦È, Tx, Ty]
% Input:
%   M_cummulative: cummulative motion estimation matrix.
% Output: 
%   M_smooth: cell of smoothed matrices.
outputArg1 = inputArg1;
outputArg2 = inputArg2;
end

