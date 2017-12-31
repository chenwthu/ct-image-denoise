function J = wavelet_denoise(I)
	wfilter = 'coif4';
	[C, S] = wavedec2(I, 3, wfilter);
	V = detcoef2('v', C, S, 1);
	sigma = median(abs(V(:))) / .6745;
	T = sigma * sqrt(2 * log(numel(I)));
	detail = prod(S(1,:))+1 : length(C);
	C(detail) = wthresh(C(detail), 's', T);
	J = real(waverec2(C, S, wfilter));
	J = J - min(J(:));
	J = J / max(J(:));
end
