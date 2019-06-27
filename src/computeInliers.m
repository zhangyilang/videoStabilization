function [ idx ] = computeInliers( transformation, points1, points2, allowedError )
%COMPUTEINLIERS Returns all indexes of the inliers for
% |transformation*points1 - points2| < allowedError

    nPoints = size(points1, 1);
    idx = [];
    for i = 1:nPoints
        p1 = points1(i, :)';
        p2 = points2(i, :)';
        p1(3) = 1;
        
        p1Trans = transformation*p1;
        p1Trans = p1Trans ./ p1Trans(3);
        p1Trans = [p1Trans(1), p1Trans(2)]';
        
        distance = p1Trans - p2;
        
        if norm(distance) < allowedError
            idx(numel(idx)+1) = i;
        end
    end
end

