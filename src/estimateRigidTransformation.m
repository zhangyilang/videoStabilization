function [ transformation, inliers ] = estimateRigidTransformation( points1, points2, nIterations, allowedError )
%ESTIMATERIGIDTRANSFORMATION estimates a ridig transformation in 2D between
%two point clouds using RANSAC
% Make sure that you have at least 3 corresponding points
%   points1: nx2 matrix of points
%   points2: nx2 matrix of points
%   nIterations: number of iterations for ransac
%   allowedError: the allowed error for to be considered an inlier


    nBest = 0;
    
    nPoints = size(points1, 1);
    sampleSize = 2;
    inliers = [];
    
    for i = 1:nIterations
        samples = randsample(nPoints, sampleSize);
        currentTransformation = computeRigidTransformation(points1(samples, :), points2(samples, :));
        currentInliers = computeInliers(currentTransformation, points1, points2, allowedError);
        if numel(currentInliers) > nBest
            nBest = numel(currentInliers);
            inliers = currentInliers;
        end
    end
    
    % recompute transformation
    transformation = computeRigidTransformation(points1(inliers, :), points2(inliers, :));
end

