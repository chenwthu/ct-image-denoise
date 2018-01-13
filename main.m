clearvars;
close all;
clc;

%%
try mex -setup C++; catch, end

%%
I = imread('teeth.bmp');
if ~ismatrix(I), I = rgb2gray(I); end
I = im2double(I);
I = imresize(I, [512, 512]);
figure('Name', 'original');
imshow(I);

%%
sigma = .05;

%%
fprintf('%-35s', '01. Gaussian filtering:');
figure('Name', 'gaussian filtering'); tic;
try
	imshow(imgaussfilt(I, sigma, 'FilterSize', 11, ...
		               'Padding', 'symmetric', ...
					   'FilterDomain', 'spatial')); toc
catch
	imshow(imfilter(I, fspecial('gaussian', 11, sigma), 'symmetric')); toc
end

%%
fprintf('%-35s', '02. Median filtering:');
figure('Name', 'median filtering'); tic;
imshow(median_filter(I, 5)); toc

%%
fprintf('%-35s', '03. Adaptive median filtering:');
figure('Name', 'adaptive median filtering'); tic;
imshow(median_filter(I, 'adaptive')); toc

%%
fprintf('%-35s', '04. Average filtering:');
figure('Name', 'average filtering'); tic;
imshow(imfilter(I, fspecial('average', 5), 'symmetric')); toc

%%
fprintf('%-35s', '05. Geometric average filtering:');
figure('Name', 'geometric average filtering'); tic;
imshow(exp(imfilter(log(I + 1), fspecial('average', 5), 'symmetric')) - 1); toc

%%
fprintf('%-35s', '06. Adaptive average filtering:');
figure('Name', 'adaptive average filtering'); tic;
imshow(adaptive_average_filter(I, sigma, 5)); toc

%%
fprintf('%-35s', '07. Wiener filtering:');
figure('Name', 'wiener filtering'); tic;
imshow(wiener2(I, [5, 5])); toc

%%
fprintf('%-35s', '08. Bilateral filtering:');
figure('Name', 'bilateral filtering'); tic;
imshow(bilateral_filter(I, 7, sigma)); toc

%%
fprintf('%-35s', '09. Guided filtering:');
figure('Name', 'guided filtering'); tic;
imshow(imguidedfilter(I)); toc

%%
fprintf('%-35s', '10. Non-local means algorithm:');
figure('Name', 'nlm denoising'); tic;
imshow(nlm(I, 3, 2, sigma, 1)); toc

%%
fprintf('%-35s', '11. Wavelet transform:');
figure('Name', 'wavelet denoising'); tic;
imshow(wavelet_denoise(I)); toc

%%
fprintf('%-35s', '12. Curvelet transform:');
figure('Name', 'curvelet denoising'); tic;
imshow(curvelet_denoise(I, sigma)); toc

%%
fprintf('%-35s', '13. Modified curvelet transform:');
figure('Name', 'modified curvelet denoising'); tic;
imshow(modified_curvelet_denoise(I, sigma)); toc

%%
try
	fprintf('%-35s', '14. Deep neural network:');
	figure('Name', 'dncnn'); tic;
	imshow(denoiseImage(I, denoisingNetwork('DnCNN'))); toc
catch
end
