function [video_new] = shift(video,M_motion)
%SHIFT 此处显示有关此函数的摘要
%   此处显示详细说明
len = length(M_motion);
video_new = video(1);
[h, w, c] = size(video{1});

max_Tx = 0;
max_Ty = 0;
min_Tx = 0;
min_Ty = 0;

sum_Tx = 0;
sum_Ty = 0;

for i = 1:len
    frame = video{i+1};
    sum_Tx = sum_Tx + round(M_motion{i}(1,3));
    sum_Ty = sum_Ty + round(M_motion{i}(2,3));   
    frame_new = zeros(h, w, c);
    
    if sum_Tx >= 0 && sum_Ty >= 0
        max_Tx = max(max_Tx, sum_Tx);
        max_Ty = max(max_Ty, sum_Ty);
        for j = 1:h-sum_Tx
            frame_new(j+sum_Tx,sum_Ty+1:w,:) = frame(j,1:w-sum_Ty,:);
        end
    end

    if sum_Tx < 0 && sum_Ty >= 0
        min_Tx = min(min_Tx, sum_Tx);
        max_Ty = max(max_Ty, sum_Ty);
        for j = -sum_Tx:h
            frame_new(j+sum_Tx+1,sum_Ty+1:w,:) = frame(j,1:w-sum_Ty,:);
        end
    end

    if sum_Tx >= 0 && sum_Ty < 0
        max_Tx = max(max_Tx, sum_Tx);
        min_Ty = min(min_Ty, sum_Ty);
        for j = 1:h-sum_Tx
            frame_new(j+sum_Tx,1:w+sum_Ty+1,:) = frame(j,-sum_Ty:w,:);
        end
    end
    
    if sum_Tx < 0 && sum_Ty < 0
        min_Tx = min(min_Tx, sum_Tx);
        min_Ty = min(min_Ty, sum_Ty);
        for j = -sum_Tx:h
            frame_new(j+sum_Tx+1,1:w+sum_Ty+1,:) = frame(j,-sum_Ty:w,:);
        end
    end
    
    video_new = [video_new; frame_new];
    
end

for i = 1:len+1
    video_new{i} = video_new{i}(1+max_Tx:h+min_Tx, 1+max_Ty:w+min_Ty, :);
end

end

