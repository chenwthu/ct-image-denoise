function J = nlm(I, r, f, sigma, h)
	try
		mex -silent nonLocalMeans.cpp;
		J = nonLocalMeans(I, r, f, sigma, h);
	catch
		J = zeros(size(I));
		I = padarray(I, [r + f, r + f], 'symmetric');
		for k = 1:size(J, 1)
			for l = 1:size(J, 2)
				s = 0;
				for u = f+k : r+r+f+k
					for v = f+l : r+r+f+l
						d = sum(sum((I(r+k : r+f+f+k, r+l : r+f+f+l) ...
							- I(u-f : u+f)) .^ 2)) / (2 * f + 1) .^ 2;
						w = exp(-max(d ^ 2 - 2 * sigma ^ 2, 0) / h ^ 2);
						J(k, l) = J(k, l) + I(u, v) * w;
						s = s + w;
					end
				end
				J(k, l) = J(k, l) / s;
			end
		end
	end
	
	J = J - min(J(:));
	J = J / max(J(:));
end
