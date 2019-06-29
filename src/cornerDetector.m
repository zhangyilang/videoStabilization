function corners = cornerDetector(method, I)
%   harrisMinEigen Compute corner metric
%   POINTS = harrisMinEigen('Harris',I) 返回角点检测的结果,
%   POINTS, 包含一个检测出一副灰度图像的角点位置的信息，采用Harris-Stephens algorithm.
%
%   POINTS = harrisMinEigen('MinEigen',I) 返回角点检测的结果,
%   POINTS, 包含一个检测出一副灰度图像的角点位置的信息, 采用由Shi and Tomasi提出的最小特征值的算法
%
%   POINTS = harrisMinEigen(...,Name,Value) 其它参数描述如下
%
%   'MinQuality'  标量 Q, 0 <= Q <= 1, 指定角的最小可接受质量，作为图像中最大角度量值的一部分。 
%    较大的Q值可用于消除错误的角点。
%    默认值: 0.01
%
%   'FilterSize'  奇数，S> = 3，指定高斯滤波器用于平滑图像的渐变。
%    滤波器的尺寸为S-by-S，滤波器的标准偏差为（S / 3）。
%    默认值: 5
%
%   'ROI' 格式为[X Y WIDTH HEIGHT]的矢量，指定一个矩形区域，其中将检测到角落。
%    [X Y]是该地区的左上角。
%    默认值: [1 1 size(I,2) size(I,1)]
% 输入图像可以是逻辑的，uint8，int16，uint16，single或
% double，它必须是真实的和非庞大的。

% 将图像转换为single类型.
I = im2single(I);

% 创建一个二维的高斯滤波器.
filterSize=5;
filter2D = createFilter(filterSize);
% 计算角度量矩阵.
metricMatrix = cornerMetric(method, I, filter2D);
% 在角度量矩阵中查找峰值，即角点.
locations = findPeaks(metricMatrix, 0.01);
locations = subPixelLocation(metricMatrix, locations);

% 在拐角位置计算拐角度量值。
metricValues = computeMetric(metricMatrix, locations);


% 将输出打包到一个角检测对象里面，以方便后续引用。
corners = cornerPoints(locations, 'Metric', metricValues);

%==========================================================================
% 通过使用计算子像素位置处的角度量值
% 双线性插值
function values = computeMetric(metric, loc)
sz = size(metric);

x = loc(:, 1);
y = loc(:, 2);
x1 = floor(x);
y1 = floor(y);
% 确保所有点都在图像边界内
x2 = min(x1 + 1, sz(2));
y2 = min(y1 + 1, sz(1));

values = metric(sub2ind(sz,y1,x1)) .* (x2-x) .* (y2-y) ...
         + metric(sub2ind(sz,y1,x2)) .* (x-x1) .* (y2-y) ...
         + metric(sub2ind(sz,y2,x1)) .* (x2-x) .* (y-y1) ...
         + metric(sub2ind(sz,y2,x2)) .* (x-x1) .* (y-y1);

%==========================================================================
% 计算角度量矩阵
function metric = cornerMetric(method, I, filter2D)
% 计算梯度
A = imfilter(I,[-1 0 1] ,'replicate','same','conv');
B = imfilter(I,[-1 0 1]','replicate','same','conv');

% 只需要有效滤波部分，因为滤波器有延迟。
A = A(2:end-1,2:end-1);
B = B(2:end-1,2:end-1);

% 计算 A, B, and C, 会在后面用来计算角度量矩阵
C = A .* B;
A = A .* A;
B = B .* B;

%  A, B, and C 经过二维高斯滤波器
A = imfilter(A,filter2D,'replicate','full','conv');
B = imfilter(B,filter2D,'replicate','full','conv');
C = imfilter(C,filter2D,'replicate','full','conv');

% 调整图像大小，去除滤波器延迟的影响
removed = max(0, (size(filter2D,1)-1) / 2 - 1);
A = A(removed+1:end-removed,removed+1:end-removed);
B = B(removed+1:end-removed,removed+1:end-removed);
C = C(removed+1:end-removed,removed+1:end-removed);
% 如果选择Harris检测方法
if strcmpi(method,'Harris')
    % 默认：k = 0.04
    k = 0.04; 
    metric = (A .* B) - (C .^ 2) - k * ( A + B ) .^ 2;
else
    metric = ((A + B) - sqrt((A - B) .^ 2 + 4 * C .^ 2)) / 2;
end

%==========================================================================
% 计算子像素位置
function loc = subPixelLocation(metric, loc)
loc = subPixelLocationImpl(metric, reshape(loc', 2, 1, []));
loc = squeeze(loc)';% 返回元素与 A 相同但删除了所有单一维度的数组 B。单一维度是指 size(A,dim) = 1 的任意维度。

%==========================================================================
% 使用双变量二次函数拟合计算子像素位置。
% Reference: http://en.wikipedia.org/wiki/Quadratic_function
function subPixelLoc = subPixelLocationImpl(metric, loc)

nLocs = size(loc,3);
patch = zeros([3, 3, nLocs], 'like', metric);
x = loc(1,1,:);
y = loc(2,1,:);
xm1 = x-1;
xp1 = x+1;
ym1 = y-1;
yp1 = y+1;
xsubs = [xm1, x, xp1;
         xm1, x, xp1;
         xm1, x, xp1];
ysubs = [ym1, ym1, ym1;
         y, y, y;
         yp1, yp1, yp1];
linind = sub2ind(size(metric), ysubs(:), xsubs(:));% 将下标转换为线性索引
patch(:) = metric(linind);

dx2 = ( patch(1,1,:) - 2*patch(1,2,:) +   patch(1,3,:) ...
    + 2*patch(2,1,:) - 4*patch(2,2,:) + 2*patch(2,3,:) ...
    +   patch(3,1,:) - 2*patch(3,2,:) +   patch(3,3,:) ) / 8;

dy2 = ( ( patch(1,1,:) + 2*patch(1,2,:) + patch(1,3,:) )...
    - 2*( patch(2,1,:) + 2*patch(2,2,:) + patch(2,3,:) )...
    +   ( patch(3,1,:) + 2*patch(3,2,:) + patch(3,3,:) )) / 8;

dxy = ( + patch(1,1,:) - patch(1,3,:) ...
        - patch(3,1,:) + patch(3,3,:) ) / 4;

dx = ( - patch(1,1,:) - 2*patch(2,1,:) - patch(3,1,:)...
       + patch(1,3,:) + 2*patch(2,3,:) + patch(3,3,:) ) / 8;

dy = ( - patch(1,1,:) - 2*patch(1,2,:) - patch(1,3,:) ...
       + patch(3,1,:) + 2*patch(3,2,:) + patch(3,3,:) ) / 8;

detinv = 1 ./ (dx2.*dy2 - 0.25.*dxy.*dxy);

% 计算峰值位置和值
x = -0.5 * (dy2.*dx - 0.5*dxy.*dy) .* detinv; % 二次峰的X偏移
y = -0.5 * (dx2.*dy - 0.5*dxy.*dx) .* detinv; % 二次峰的Y偏移

% 如果两个偏移都小于1个像素，则子像素位置为
% 被视为有效
isValid = (abs(x) < 1) & (abs(y) < 1);
x(~isValid) = 0;
y(~isValid) = 0;
subPixelLoc = [x; y] + loc;

%==========================================================================
% 创建一个高斯滤波器
function f = createFilter(filterSize)
sigma = filterSize / 3;
f = fspecial('gaussian', filterSize, sigma);




