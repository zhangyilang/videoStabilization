function tf = isValidPointObj(points)
% isValidPointObj returns true if points are a MSERRegions or a
% FeaturePoints object

%#codegen
tf = isa(points, 'MSERRegions') || ...
    isa(points, 'vision.internal.MSERRegions_cg') || ...
    isa(points, 'vision.internal.FeaturePointsImpl');
