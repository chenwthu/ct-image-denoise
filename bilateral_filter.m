function J = bilateral_filter(I, r, sigma)
	[X, Y] = meshgrid(-r:r, -r:r);
	D = exp(-(X .^ 2 + Y .^ 2) / (2 * sigma ^ 2));
	[m, n] = size(I);
	J = zeros(m, n);
	for k = 1:m
		for l = 1:n
			patchLeft = max(1, k - r);
			patchRight = min(m, k + r);
			patchTop = max(1, l - r);
			patchBottom = min(n, l + r);
			patch = I(patchLeft:patchRight, patchTop:patchBottom);
			R = exp(-(I(k, l) - patch) .^ 2 / (2 * sigma ^ 2));
			W = D((patchLeft:patchRight) - k + r + 1, ...
				(patchTop:patchBottom) - l + r + 1) .* R;
			J(k, l) = sum(W(:) .* patch(:)) / sum(W(:));
		end
	end
	J = J - min(J(:));
	J = J / max(J(:));
end