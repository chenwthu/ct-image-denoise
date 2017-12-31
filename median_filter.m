function J = median_filter(I, hsize)
	padopt = 'symmetric';
	if strcmp(hsize, 'adaptive')
		try
			mex -silent adaptiveMedianFilter.cpp;
			J = adaptiveMedianFilter(I);
		catch
			J = I; J(:) = 0;
			done = false(size(I));
			max_hsize = min(size(I) / 20);
			for hsize = 3:2:max_hsize
				low = ordfilt2(I, 1, ones(hsize, hsize), padopt);
				high = ordfilt2(I, hsize ^ 2, ones(hsize, hsize), padopt);
				med = medfilt2(I, [hsize, hsize], padopt);

				output = (low < med) & (med < high) & (~done);
				if ~any(output(:)), continue; end
				is_noise = ~((low < I) & (I < high));
				J(is_noise) = med(is_noise);
				J(~is_noise) = I(~is_noise);

				done = done | output;
				if all(done(:)), break; end
			end
			J(~done) = med(~done);
		end
	elseif isnumeric(hsize)
		if all(hsize > 0) && all(~isinf(hsize))
			if numel(hsize) == 1
				J = medfilt2(I, [hsize, hsize], padopt);
			elseif numel(hsize) == 2
				J = medfilt2(I, hsize, padopt);
			end
		else
			J = I;
		end
	else
		J = I;
	end
end
