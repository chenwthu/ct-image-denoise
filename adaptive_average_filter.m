function J = adaptive_average_filter(I, sigma, hsize)
	[M, N] = size(I);
	J = zeros(M * N, hsize ^ 2 + 1);
	J(:, 1) = I(:);
	I = padarray(I, (hsize - 1) / 2 * [1, 1]);
	for k = 1:hsize
		for l = 1:hsize
			L = I(k : k+M-1, l : l+N-1);
			J(:, (k - 1) * hsize + l + 1) = L(:);
		end
	end
	J = reshape(J(:, 1) - sigma ^ 2 * (J(:, 1) - mean(J(:, 2:end), 2)) ...
		./ max(var(J(:, 2:end), 0, 2), sigma ^ 2), M, N);
	J = J - min(J(:));
	J = J / max(J(:));
end
