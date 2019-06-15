function [output] = resample(input,prob)
%SAMPLE
%   Resample the input vector accroding to its probability. Resample
%   dosen't change the total number of data.

len = length(input);
output = zeros(1,len);
sumProb = zeros(1,len);

tmp = 0;
for i = 1:len
    tmp = tmp + prob(i);
    sumProb(i) = tmp;
end

for i = 1:len
    randNumber = rand * sumProb(end);
    for j = 1:len
        if randNumber <= sumProb(j)
            break
        end
    end
    output(i) = input(j);
end

end

