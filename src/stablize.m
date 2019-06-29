function stableimg = stablize(video, M_smooth)
stableimg = cell(length(video),1);
M = eye(3,3);
for i = 1:length(video)
    sizeVideo = size(video{i});
%     stableimg{i} = zeros(sizeVideo(1),sizeVideo(2),sizeVideo(3));
    stableimg{i} = zeros(ceil(2 * sizeVideo(1)),ceil(2 * sizeVideo(2)),sizeVideo(3));
    M = (M_smooth{i}^-1);
%     M1 = M^-1;
    for j = 1:sizeVideo(1)
        array = [j * ones(1,sizeVideo(2)); 1:sizeVideo(2);ones(1,sizeVideo(2))];
        M(1,1) = 1;
        M(1,2) = 0;
        M(2,1) = 0;
        M(2,2) = 1;
        newIndex = M * array;
%         new_x = ceil(newIndex(1,:) + 0.5 * sizeVideo(1));
%         new_y = ceil(newIndex(2,:) + 0.5 * sizeVideo(2));
%         stableimg{i}(new_x,new_y,:) = video{i}(j * ones(1,sizeVideo(2)),1:sizeVideo(2),:);
        new_x = ceil(newIndex(1,:));
        new_y = ceil(newIndex(2,:));
        new_x(new_y <= 0) = 0;
        new_y(new_x <= 0) = 0;
        validate = find(new_y > 0);
        stableimg{i}(new_x(validate),new_y(validate),:) = video{i}(j * ones(1,length(validate)),validate,:);
        
%         for k = 1:size(video{i},2)
%             temp = M*[j k 1]';
%             newIndex=ceil(temp(1) + 0.5 * sizeVideo(1));
%             newIndey=ceil(temp(2) + 0.5 * sizeVideo(2));
%             stableimg{i}(newIndex,newIndey,:) = video{i}(j,k,:);
%         end
    end
end
end