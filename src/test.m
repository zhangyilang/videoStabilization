%% test for Kalman filtering
M_test = cell(1,99);
y_test = ones(1,99);
y_noise = zeros(1,99);
for i = 1:99
    M_test{i} = sqrt(1/2) + 1e-2 * randn(3);
    y_noise(i) = sqrt(M_test{i}(1,1)^2 + M_test{i}(2,1)^2);
end

M_test_smooth = smooth_Kalman(M_test);

y_test_smooth = zeros(1,99);
for i = 2:100
    y_test_smooth(i-1) = sqrt(M_test_smooth{i}(1,1)^2 + M_test_smooth{i}(2,1)^2);
end
% y_test_smooth = fft(y_test_smooth);
% y_test_smooth(10:end) = 0;
% y_test_smooth = real(ifft(y_test_smooth));

figure(1)
hold on
plot(1:99, y_noise, 'r')
plot(1:99, y_test, 'b')
plot(1:99, y_test_smooth, 'g')
legend('add noise', 'origin', 'Kalman filter')

%% test for particle filtering
M_test = cell(1,99);
y_test = ones(1,99);
y_noise = zeros(1,99);
for i = 1:99
    M_test{i} = sqrt(1/2) + 1e-2 * randn(3);
    y_noise(i) = sqrt(M_test{i}(1,1)^2 + M_test{i}(2,1)^2);
end

M_test_smooth = smooth_particle(M_test,2000);

y_test_smooth = zeros(1,99);
for i = 2:100
    y_test_smooth(i-1) = sqrt(M_test_smooth{i}(1,1)^2 + M_test_smooth{i}(2,1)^2);
end
% y_test_smooth = fft(y_test_smooth);
% y_test_smooth(10:end) = 0;
% y_test_smooth = real(ifft(y_test_smooth));

figure(2)
hold on
plot(1:99, y_noise, 'r')
plot(1:99, y_test, 'b')
plot(1:99, y_test_smooth, 'g')
legend('add noise', 'origin', 'particle filter')