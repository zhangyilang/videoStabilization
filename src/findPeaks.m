function loc = findPeaks(metric,quality)
% 返回图像度量标准中的局部最大值。 质量因子是一个门槛
% 表示为度量最大值的一部分。

maxMetric = max(metric(:));
if maxMetric <= eps(0)                  % 4.9407e-324
    loc = zeros(0,2, 'single');
else
    
    bw = imregionalmax(metric, 8);      % 返回二进制图像BW该 BW 标识灰度图像I中的区域最大值。区域最小值是具有恒定强度值的像素的连接组件,周围环绕着值较低的像素。
    
    threshold = quality * maxMetric;    % 默认为0.01
    bw(metric < threshold) = 0;
    bw = bwmorph(bw, 'shrink', Inf);    % 删除内部像素以留下形状的轮廓。
    
    % 排除边界上的点
    bw(1, :) = 0;
    bw(end, :) = 0;
    bw(:, 1) = 0;
    bw(:, end) = 0;
    
    % 找到峰的位置
    idx = find(bw);                     % 查找非零元素的索引和值
    loc = zeros([length(idx) 2], 'like', metric );
    [loc(:, 2), loc(:, 1)] = ind2sub(size(metric), idx);
end