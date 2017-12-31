clearvars;
close all;
clc;

%%
% try mex -setup cpp; catch, end

%%
I = imread('teeth.bmp');
if ~ismatrix(I), I = rgb2gray(I); end
I = im2double(I);
I = imresize(I, [640, 640]);
figure('Name', 'original');
imshow(I);

%%
sigma = .05;

%%
fprintf('%-30s', 'Gaussian filtering:');
figure('Name', 'gaussian filtering'); tic;
try
	imshow(imgaussfilt(I, sigma, 'Padding', 'symmetric')); toc
catch
	imshow(imfilter(I, fspecial('gaussian', 13, sigma), 'symmetric')); toc
end

%%
fprintf('%-30s', 'Median filtering:');
figure('Name', 'median filtering'); tic;
imshow(median_filter(I, 5)); toc

%%
fprintf('%-30s', 'Adaptive Median filtering:');
figure('Name', 'adaptive median filtering'); tic;
imshow(median_filter(I, 'adaptive')); toc

%%
fprintf('%-30s', 'Wiener filtering:');
figure('Name', 'wiener filtering'); tic;
imshow(wiener2(I, [5, 5])); toc

%%
fprintf('%-30s', 'Bilateral filtering:');
figure('Name', 'bilateral filtering'); tic;
imshow(bilateral_filter(I, 7, sigma)); toc

%%
fprintf('%-30s', 'Non-local means algorithm:');
figure('Name', 'nlm denoising'); tic;
imshow(nlm(I, 3, 2, sigma, 1)); toc

%%
fprintf('%-30s', 'Wavelet transform:');
figure('Name', 'wavelet denoising'); tic;
imshow(wavelet_denoise(I)); toc

%%
fprintf('%-30s', 'Curvelet transform:');
figure('Name', 'curvelet denoising'); tic;
imshow(curvelet_denoise(I, sigma)); toc

%%
fprintf('%-30s', 'Modified Curvelet transform:');
figure('Name', 'modified curvelet denoising'); tic;
imshow(modified_curvelet_denoise(I, sigma)); toc

%%
% try
% 	fprintf('%-30s', 'Deep neural network:');
% 	figure('Name', 'dncnn'); tic;
% 	imshow(denoiseImage(I, denoisingNetwork('DnCNN'))); toc
% catch
% end
