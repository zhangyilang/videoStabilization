function [M_motion] = removeDC(M_motion_DC)
%REMOVEDC 此处显示有关此函数的摘要
%   此处显示详细说明
len = length(M_motion_DC);
Tx_sum = 0;
Ty_sum = 0;
for i = 1:len
    Tx_sum = Tx_sum + M_motion_DC{i}(1,3);
    Ty_sum = Ty_sum + M_motion_DC{i}(2,3);
end

Tx_mean = Tx_sum / len;
Ty_mean = Ty_sum / len;

M_motion = M_motion_DC;
for i = 1:len
    M_motion{i}(1,3) = M_motion{i}(1,3) - Tx_mean;
    M_motion{i}(2,3) = M_motion{i}(2,3) - Ty_mean;
end
end
