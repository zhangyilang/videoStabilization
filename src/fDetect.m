function [ output_mat ] = fDetect( img )
%FDETECT Summary of this function goes here
%   Detailed explanation goes here
[l, b] = size(img);

% Harris threshold
%threshold_h = 5e+10;
% Shi-Tomasi threshold
threshold_st = 5e+3;
% k-value for Harris
%k = 0.05;

% Sobel operators to compute partial derivative w.r.t. x and y
Gx = [-1 0 1; -2 0 2; -1 0 1];
Gy = [-1 -2 -1; 0 0 0; 1 2 1];

% Initialize Gradient Matrices
Ix = zeros(l, b);
Iy = zeros(l, b);

% Zero padding
img = [zeros(size(img, 1), 1) img zeros(size(img, 1), 1)];
img = [zeros(1, size(img, 2)); img; zeros(1, size(img, 2))];

% Convolve to calculate Intensity Gradient
for i = 1:(size(img, 1)-2)
    for j = 1:(size(img, 2)-2)
        Ix(i, j) = sum(sum(img(i:i+2, j:j+2).*Gx));
        Iy(i, j) = sum(sum(img(i:i+2, j:j+2).*Gy));
    end
end

Ix_2 = Ix.^2;
Iy_2 = Iy.^2;
Ixy = Ix.*Iy;
[m, n] = size(Ix);

% Initialize Feature matrices
%F_h = zeros(m, n);
F_st = zeros(m, n);

% Calculate R-value and check if greater than threshold
for i = 1:m
    for j = 1:n
        % Gradient Matrix
        M = [Ix_2(i, j) Ixy(i, j);Ixy(i, j) Iy_2(i, j)];
        % Harris R-value
        %R_h = abs(det(M) - k*(trace(M)^2));
        % Shi-Tomasi R-value
        R_st = min([M(1, 1) M(2, 2)]);
        % Check for features
        %if R_h > threshold_h
        %    F_h(i, j) = 1;
        %end
        if R_st > threshold_st
            F_st(i, j) = 1;
        end
    end
end
output_mat = F_st;
end

