function J = curvelet_denoise(I, sigma)
	C = fdct(fftshift(ifft2(ones(size(I)))) * sqrt(numel(I)), false);
	
	E = cell(size(C));
	for u = 1:length(C)
		E{u} = cell(size(C{u}));
		for v = 1:length(C{u})
			A = C{u}{v};
			E{u}{v} = sqrt(sum(sum(A .* conj(A))) / numel(A));
		end
	end
	
	C = fdct(I, true);
	for u = 2:length(C)
		T = (3 + (u == length(C))) * sigma;
		for v = 1:length(C{u})
			C{u}{v} = C{u}{v} .* (abs(C{u}{v}) > T * E{u}{v});
		end
	end
	
	J = real(ifdct(C, true));
	J = J - min(J(:));
	J = J / max(J(:));
end

function Y = fdct(X, is_real)
	X = fftshift(fft2(ifftshift(X))) / sqrt(numel(X));
	[M, N] = size(X);
	n_scales = ceil(log2(min(M, N)) - 3);
	n_angles = [1, 16 * 2 .^ ceil((0 : n_scales-2) / 2)];
	n_angles(n_scales) = 1;
	Y = cell(1, n_scales);
	for k = 1:n_scales, Y{k} = cell(1, n_angles(k)); end
	
	m = M / 6;
	n = N / 6;
	w_len = floor(2 * [m, n]) - floor([m, n]) - 1;
	[wl_1, wr_1] = fdct_window(0 : 1/w_len(1) : 1);
	[wl_2, wr_2] = fdct_window(0 : 1/w_len(2) : 1);
	lowpass = [wl_1, ones(1, 2 * floor(m) + 1), wr_1]' ...
		* [wl_2, ones(1, 2 * floor(n) + 1), wr_2];
	highpass = sqrt(1 - lowpass .^ 2);
	id_1 = (-floor(2 * m) : floor(2 * m)) + ceil((M + 1) / 2);
	id_2 = (-floor(2 * n) : floor(2 * n)) + ceil((N + 1) / 2);
	X_low = X(id_1, id_2) .* lowpass;
	X_high = X; X_high(id_1, id_2) = X(id_1, id_2) .* highpass;
	Y{end}{1} = fftshift(ifft2(ifftshift(X_high))) * sqrt(M * N);
	if is_real, Y{end}{1} = real(Y{end}{1}); end
	
	for k = n_scales-1 : -1 : 2
		m = m / 2;
		n = n / 2;
		w_len = floor(2 * [m, n]) - floor([m, n]) - 1;
		[wl_1, wr_1] = fdct_window(0 : 1/w_len(1) : 1);
		[wl_2, wr_2] = fdct_window(0 : 1/w_len(2) : 1);
		lowpass = [wl_1, ones(1, 2 * floor(m) + 1), wr_1]' ...
			* [wl_2, ones(1, 2 * floor(n) + 1), wr_2];
		highpass = sqrt(1 - lowpass .^ 2);
		id_1 = (-floor(2 * m) : floor(2 * m)) + floor(4 * m) + 1;
		id_2 = (-floor(2 * n) : floor(2 * n)) + floor(4 * n) + 1;
		X_high = X_low; X_low = X_low(id_1, id_2);
		X_high(id_1, id_2) = X_low .* highpass;
		X_low = X_low .* lowpass;
		
		if is_real, n_quad = 2; else, n_quad = 4; end
		n_quad_angles = n_angles(k) / 4;
		l = 0;
		for q = 1:n_quad
			m_x = m * (mod(q, 2) == 0) + n * (mod(q, 2) == 1);
			m_y = m * (mod(q, 2) == 1) + n * (mod(q, 2) == 0);
			wl = round((0: .5/n_quad_angles : .5) * 2 * floor(4 * m_x) + 1);
			wr = 2 * floor(4 * m_x) - wl + 2;
			w = [wl, wr(end+mod(n_quad_angles,2)-1 : -1 : 1)];
			w_end = w(2 : 2 : end-1);
			w_mid = (w_end(1 : end-1) + w_end(2:end)) / 2;
			
			l = l + 1;
			first_w_end_y = round(floor(4 * m_y) / n_quad_angles + 1);
			w_len_corner = floor(4 * m_y) - floor(m_y) + ceil(first_w_end_y / 4);
			y_corner = 1:w_len_corner;
			[x, y] = meshgrid(1 : 2*floor(4*m_x)+1, y_corner);
			w_width = w_end(1) + w_end(2) - 1;
			w_slope = (floor(4 * m_x) + 1 - w_end(1)) / floor(4 * m_y);
			line_l = round(2 - w_end(1) + w_slope * (y_corner - 1));
			[data, data_x, data_y] = deal(zeros(w_len_corner, w_width));
			first_row = floor(4 * m_y) + 2 - ceil((w_len_corner + 1) / 2) ...
				+ mod(w_len_corner + 1, 2) * (mod(q - 2, 2) == q - 2);
			first_col = floor(4 * m_x) + 2 - ceil((w_width + 1) / 2) ...
				+ mod(w_width + 1, 2) * (mod(q - 3, 2) == q - 3);
			for row = y_corner
				cols = line_l(row) ...
					+ mod((0 : w_width-1) - line_l(row) + first_col, w_width);
				a_cols = round((cols + 1 + abs(cols - 1)) / 2);
				new_row = 1 + mod(row - first_row, w_len_corner);
				data(new_row, :) = X_high(row, a_cols) .* (cols > 0);
				data_x(new_row, :) = x(row, a_cols);
				data_y(new_row, :) = y(row, a_cols);
			end
			w_slope_r = (floor(4 * m_x) + 1 - w_mid(1)) / floor(4 * m_y);
			line_rm = w_mid(1) + w_slope_r * (data_y - 1);
			C = 1 / (1 / (2 * floor(4 * m_x) / (w_end(1) - 1) - 1) ...
				+ 1 / (2 * floor(4 * m_y) / (first_w_end_y - 1) - 1));
			id = (data_x - 1) / floor(4 * m_x) + (data_y - 1) / floor(4 * m_y);
			data_x(id == 2) = data_x(id == 2) + 1;
			id_1 = (data_x - 1) / floor(4 * m_x) + (data_y - 1) / floor(4 * m_y);
			id_2 = (data_x - 1) / floor(4 * m_x) - (data_y - 1) / floor(4 * m_y);
			wll = fdct_window(C * id_2 ./ (2 - id_1) ...
				+ C / (2 * floor(4 * m_y) / (first_w_end_y - 1) - 1));
			[~, wrr] = fdct_window(floor(4 * m_y) / (w_end(2) - w_end(1)) ...
				* (data_x - line_rm) ./ (floor(4 * m_y) + 1 - data_y) + .5);
			data = rot90(data .* wll .* wrr, 1 - q);
			c = fftshift(ifft2(ifftshift(data))) * sqrt(numel(data));
			if is_real
				Y{k}{l} = sqrt(2) * real(c);
				Y{k}{l + n_angles(k) / 2} = sqrt(2) * imag(c);
			else
				Y{k}{l} = c;
			end
			
			w_len = floor(4 * m_y) - floor(m_y);
			first_row = floor(4 * m_y) + 2 - ceil((w_len + 1) / 2) ...
				+ mod(w_len + 1, 2) * (mod(q - 2, 2) == q - 2);
			for subl = 2 : n_quad_angles-1
				l = l + 1;
				w_width = w_end(subl + 1) - w_end(subl - 1) + 1;
				w_slope = (floor(4 * m_x) + 1 - w_end(subl)) / floor(4 * m_y);
				line_l = round(w_end(subl - 1) + w_slope * (0 : w_len-1));
				[data, data_x, data_y] = deal(zeros(w_len, w_width));
				first_col = floor(4 * m_x) + 2 - ceil((w_width + 1) / 2) ...
					+ mod(w_width + 1, 2) * (mod(q - 3, 2) == q - 3);
				for row = 1:w_len
					cols = line_l(row) ...
						+ mod((0 : w_width-1) - line_l(row) + first_col, w_width);
					new_row = 1 + mod(row - first_row, w_len);
					data(new_row, :) = X_high(row, cols);
					data_x(new_row, :) = x(row, cols);
					data_y(new_row, :) = y(row, cols);
				end
				w_slope_l = (floor(4 * m_x) + 1 - w_mid(subl - 1)) / floor(4 * m_y);
				line_lm = w_mid(subl - 1) + w_slope_l * (data_y - 1);
				w_slope_r = (floor(4 * m_x) + 1 - w_mid(subl)) / floor(4 * m_y);
				line_rm = w_mid(subl) + w_slope_r * (data_y - 1);
				wll = fdct_window( ...
					.5 + floor(4 * m_y) / (w_end(subl) - w_end(subl - 1)) ...
					* (data_x - line_lm) ./ (floor(4 * m_y) + 1 - data_y));
				[~, wrr] = fdct_window( ...
					.5 + floor(4 * m_y) / (w_end(subl + 1) - w_end(subl)) ...
					* (data_x - line_rm) ./ (floor(4 * m_y) + 1 - data_y));
				data = rot90(data .* wll .* wrr, 1 - q);
				c = fftshift(ifft2(ifftshift(data))) * sqrt(numel(data));
				if is_real
					Y{k}{l} = sqrt(2) * real(c);
					Y{k}{l + n_angles(k) / 2} = sqrt(2) * imag(c);
				else
					Y{k}{l} = c;
				end
			end
			
			l = l + 1;
			w_width = 4 * floor(4 * m_x) + 3 - w_end(end) - w_end(end - 1);
			w_slope = (floor(4 * m_x) + 1 - w_end(end)) / floor(4 * m_y);
			line_l = round(w_end(end - 1) + w_slope * (y_corner - 1));
			[data, data_x, data_y] = deal(zeros(w_len_corner, w_width));
			first_row = floor(4 * m_y) + 2 - ceil((w_len_corner + 1) / 2) ...
				+ mod(w_len_corner + 1, 2) * (mod(q - 2, 2) == q - 2);
			first_col = floor(4 * m_x) + 2 - ceil((w_width + 1) / 2) ...
				+ mod(w_width + 1, 2) * (mod(q - 3, 2) == q - 3);
			for row = y_corner
				cols = line_l(row) ...
					+ mod((0 : w_width-1) - line_l(row) + first_col, w_width);
				a_cols = round((cols + 2 * floor(4 * m_x) + 1 ...
					- abs(cols - 2 * floor(4 * m_x) - 1)) / 2);
				new_row = 1 + mod(row - first_row, w_len_corner);
				data(new_row, :) = X_high(row, a_cols) ...
					.* (cols <= 2 * floor(4 * m_x) + 1);
				data_x(new_row, :) = x(row, a_cols);
				data_y(new_row, :) = y(row, a_cols);
			end
			w_slope_l = (floor(4 * m_x) + 1 - w_mid(end)) / floor(4 * m_y);
			line_lm = w_mid(end) + w_slope_l * (data_y - 1);
			C = -1 / (2 * floor(4 * m_x) / (w_end(end) - 1) - 1 ...
				+ 1 / (2 * floor(4 * m_y) / (first_w_end_y - 1) - 1));
			id = (data_x - 1) / floor(4 * m_x) == (data_y - 1) / floor(4 * m_y);
			data_x(id) = data_x(id) - 1;
			id_1 = (data_x - 1) / floor(4 * m_x) + (data_y - 1) / floor(4 * m_y);
			id_2 = (data_x - 1) / floor(4 * m_x) - (data_y - 1) / floor(4 * m_y);
			wll = fdct_window( ...
				.5 + floor(4 * m_y) / (w_end(end) - w_end(end - 1)) ...
				* (data_x - line_lm) ./ (floor(4 * m_y) + 1 - data_y));
			[~, wrr] = fdct_window(C * (2 - id_1) ./ id_2 ...
				- C * (2 * floor(4 * m_x) / (w_end(end) - 1) - 1));
			data = rot90(data .* wll .* wrr, 1 - q);
			c = fftshift(ifft2(ifftshift(data))) * sqrt(numel(data));
			if is_real
				Y{k}{l} = sqrt(2) * real(c);
				Y{k}{l + n_angles(k) / 2} = sqrt(2) * imag(c);
			else
				Y{k}{l} = c;
			end
			
			if q < n_quad, X_high = rot90(X_high); end
		end
	end
	
	Y{1}{1} = fftshift(ifft2(ifftshift(X_low))) * sqrt(numel(X_low));
	if is_real, Y{1}{1} = real(Y{1}{1}); end
end

function X = ifdct(Y, is_real)
	[M, N] = size(Y{end}{1});
	n_scales = length(Y);
	n_angles = [1, length(Y{2}) .* 2 .^ ceil((0 : n_scales-2) / 2)];
	n_angles(n_scales) = 1;
	
	m = M / 6;
	n = N / 6;
	X = zeros(2 * floor(2 * m) + 1, 2 * floor(2 * n) + 1);
	w_len = floor(2 * [m, n]) - floor([m, n]) - 1;
	[wl_1, wr_1] = fdct_window(0 : 1/w_len(1) : 1);
	[wl_2, wr_2] = fdct_window(0 : 1/w_len(2) : 1);
	lowpass = [wl_1, ones(1, 2 * floor(m) + 1), wr_1]' ...
		* [wl_2, ones(1, 2 * floor(n) + 1), wr_2];
	highpass = sqrt(1 - lowpass .^ 2);
	
	X_ul = [1, 1];
	for k = n_scales-1 : -1 : 2
		m = m / 2;
		n = n / 2;
		w_len = floor(2 * [m, n]) - floor([m, n]) - 1;
		[wl_1, wr_1] = fdct_window(0 : 1/w_len(1) : 1);
		[wl_2, wr_2] = fdct_window(0 : 1/w_len(2) : 1);
		lowpass_next = [wl_1, ones(1, 2 * floor(m) + 1), wr_1]' ...
			* [wl_2, ones(1, 2 * floor(n) + 1), wr_2];
		highpass_next = sqrt(1 - lowpass_next .^ 2);
		Xk = zeros(2 * floor(4 * m) + 1, 2 * floor(4 * n) + 1);
		
		if is_real, n_quad = 2; else, n_quad = 4; end
		n_quad_angles = n_angles(k) / 4;
		l = 0;
		for q = 1:n_quad
			m_x = m * (mod(q, 2) == 0) + n * (mod(q, 2) == 1);
			m_y = m * (mod(q, 2) == 1) + n * (mod(q, 2) == 0);
			wl = round((0 : .5/n_quad_angles : .5) * 2 * floor(4 * m_x) + 1);
			wr = 2 * floor(4 * m_x) - wl + 2;
			w = [wl, wr(end+mod(n_quad_angles,2)-1 : -1 : 1)];
			w_end = w(2 : 2 : end-1);
			w_mid = (w_end(1 : end-1) + w_end(2:end)) / 2;
			
			l = l + 1;
			first_w_end_y = round(floor(4 * m_y) / n_quad_angles + 1);
			w_len_corner = floor(4 * m_y) - floor(m_y) + ceil(first_w_end_y / 4);
			y_corner = 1:w_len_corner;
			[x, y] = meshgrid(1 : 2*floor(4*m_x)+1, y_corner);
			w_width = w_end(1) + w_end(2) - 1;
			w_slope = (floor(4 * m_x) + 1 - w_end(1)) / floor(4 * m_y);
			line_l = round(2 - w_end(1) + w_slope * (y_corner - 1));
			[data_x, data_y] = deal(zeros(w_len_corner, w_width));
			first_row = floor(4 * m_y) + 2 - ceil((w_len_corner + 1) / 2) ...
				+ mod(w_len_corner + 1, 2) * (mod(q - 2, 2) == q - 2);
			first_col = floor(4 * m_x) + 2 - ceil((w_width + 1) / 2) ...
				+ mod(w_width + 1, 2) * (mod(q - 3, 2) == q - 3);
			for row = y_corner
				cols = line_l(row) ...
					+ mod((0 : w_width-1) - line_l(row) + first_col, w_width);
				a_cols = round((cols + 1 + abs(cols - 1)) / 2);
				new_row = 1 + mod(row - first_row, w_len_corner);
				data_x(new_row, :) = x(row, a_cols);
				data_y(new_row, :) = y(row, a_cols);
			end
			w_slope_r = (floor(4 * m_x) + 1 - w_mid(1)) / floor(4 * m_y);
			line_rm = w_mid(1) + w_slope_r * (data_y - 1);
			C = 1 / (1 / (2 * floor(4 * m_x) / (w_end(1) - 1) - 1) ...
				+ 1 / (2 * floor(4 * m_y) / (first_w_end_y - 1) - 1));
			id = (data_x - 1) / floor(4 * m_x) + (data_y - 1) / floor(4 * m_y);
			data_x(id == 2) = data_x(id == 2) + 1;
			id_1 = (data_x - 1) / floor(4 * m_x) + (data_y - 1) / floor(4 * m_y);
			id_2 = (data_x - 1) / floor(4 * m_x) - (data_y - 1) / floor(4 * m_y);
			wll = fdct_window(C * id_2 ./ (2 - id_1) ...
				+ C / (2 * floor(4 * m_y) / (first_w_end_y - 1) - 1));
			[~, wrr] = fdct_window(floor(4 * m_y) / (w_end(2) - w_end(1)) ...
				* (data_x - line_rm) ./ (floor(4 * m_y) + 1 - data_y) + .5);
			if is_real
				c = Y{k}{l} + 1i * Y{k}{l + n_angles(k) / 2};
				data = fftshift(fft2(ifftshift(c))) / sqrt(numel(c)) / sqrt(2);
			else
				data = fftshift(fft2(ifftshift(Y{k}{l}))) / sqrt(numel(Y{k}{l}));
			end
			data = rot90(data, q - 1) .* wll .* wrr;
			for row = y_corner
				cols = line_l(row) ...
					+ mod((0 : w_width-1) - line_l(row) + first_col, w_width);
				a_cols = round((cols + 1 + abs(cols - 1)) / 2);
				new_row = 1 + mod(row - first_row, w_len_corner);
				Xk(row, a_cols) = Xk(row, a_cols) + data(new_row, :);
			end
			
			w_len = floor(4 * m_y) - floor(m_y);
			first_row = floor(4 * m_y) + 2 - ceil((w_len + 1) / 2) ...
				+ mod(w_len + 1, 2) * (mod(q - 2, 2) == q - 2);
			for subl = 2 : n_quad_angles-1
				l = l + 1;
				w_width = w_end(subl + 1) - w_end(subl - 1) + 1;
				w_slope = (floor(4 * m_x) + 1 - w_end(subl)) / floor(4 * m_y);
				line_l = round(w_end(subl - 1) + w_slope * (0 : w_len-1));
				[data_x, data_y] = deal(zeros(w_len, w_width));
				first_col = floor(4 * m_x) + 2 - ceil((w_width + 1) / 2) ...
					+ mod(w_width + 1, 2) * (mod(q - 3, 2) == q - 3);
				for row = 1:w_len
					cols = line_l(row) ...
						+ mod((0 : w_width-1) - line_l(row) + first_col, w_width);
					new_row = 1 + mod(row - first_row, w_len);
					data_x(new_row, :) = x(row, cols);
					data_y(new_row, :) = y(row, cols);
				end
				w_slope_l = (floor(4 * m_x) + 1 - w_mid(subl - 1)) / floor(4 * m_y);
				line_lm = w_mid(subl - 1) + w_slope_l * (data_y - 1);
				w_slope_r = (floor(4 * m_x) + 1 - w_mid(subl)) / floor(4 * m_y);
				line_rm = w_mid(subl) + w_slope_r * (data_y - 1);
				wll = fdct_window( ...
					.5 + floor(4 * m_y) / (w_end(subl) - w_end(subl - 1)) ...
					* (data_x - line_lm) ./ (floor(4 * m_y) + 1 - data_y));
				[~, wrr] = fdct_window( ...
					.5 + floor(4 * m_y) / (w_end(subl + 1) - w_end(subl)) ...
					* (data_x - line_rm) ./ (floor(4 * m_y) + 1 - data_y));
				if is_real
					c = Y{k}{l} + 1i * Y{k}{l + n_angles(k) / 2};
					data = fftshift(fft2(ifftshift(c))) / sqrt(numel(c)) / sqrt(2);
				else
					data = fftshift(fft2(ifftshift(Y{k}{l}))) / sqrt(numel(Y{k}{l}));
				end
				data = rot90(data, q - 1) .* wll .* wrr;
				for row = 1:w_len
					cols = line_l(row) ...
						+ mod((0 : w_width-1) - line_l(row) + first_col, w_width);
					new_row = 1 + mod(row - first_row, w_len);
					Xk(row, cols) = Xk(row, cols) + data(new_row, :);
				end
			end
			
			l = l + 1;
			w_width = 4 * floor(4 * m_x) + 3 - w_end(end) - w_end(end - 1);
			w_slope = (floor(4 * m_x) + 1 - w_end(end)) / floor(4 * m_y);
			line_l = round(w_end(end - 1) + w_slope * (y_corner - 1));
			[data_x, data_y] = deal(zeros(w_len_corner, w_width));
			first_row = floor(4 * m_y) + 2 - ceil((w_len_corner + 1) / 2) ...
				+ mod(w_len_corner + 1, 2) * (mod(q - 2, 2) == q - 2);
			first_col = floor(4 * m_x) + 2 - ceil((w_width + 1) / 2) ...
				+ mod(w_width + 1, 2) * (mod(q - 3, 2) == q - 3);
			for row = y_corner
				cols = line_l(row) ...
					+ mod((0 : w_width-1) - line_l(row) + first_col, w_width);
				a_cols = round((cols + 2 * floor(4 * m_x) + 1 ...
					- abs(cols - 2 * floor(4 * m_x) - 1)) / 2);
				new_row = 1 + mod(row - first_row, w_len_corner);
				data_x(new_row, :) = x(row, a_cols);
				data_y(new_row, :) = y(row, a_cols);
			end
			y = y_corner' * ones(1, w_width);
			w_slope_l = (floor(4 * m_x) + 1 - w_mid(end)) / floor(4 * m_y);
			line_lm = w_mid(end) + w_slope_l * (data_y - 1);
			C = -1 / (2 * floor(4 * m_x) / (w_end(end) - 1) - 1 ...
				+ 1 / (2 * floor(4 * m_y) / (first_w_end_y - 1) - 1));
			id = (data_x - 1) / floor(4 * m_x) == (data_y - 1) / floor(4 * m_y);
			data_x(id) = data_x(id) - 1;
			id_1 = (data_x - 1) / floor(4 * m_x) + (data_y - 1) / floor(4 * m_y);
			id_2 = (data_x - 1) / floor(4 * m_x) - (data_y - 1) / floor(4 * m_y);
			wll = fdct_window( ...
				.5 + floor(4 * m_y) / (w_end(end) - w_end(end - 1)) ...
				* (data_x - line_lm) ./ (floor(4 * m_y) + 1 - data_y));
			[~, wrr] = fdct_window(C * (2 - id_1) ./ id_2 ...
				- C * (2 * floor(4 * m_x) / (w_end(end) - 1) - 1));
			if is_real
				c = Y{k}{l} + 1i * Y{k}{l + n_angles(k) / 2};
				data = fftshift(fft2(ifftshift(c))) / sqrt(numel(c)) / sqrt(2);
			else
				data = fftshift(fft2(ifftshift(Y{k}{l}))) / sqrt(numel(Y{k}{l}));
			end
			data = rot90(data, q - 1) .* wll .* wrr;
			for row = y_corner
				cols = line_l(row) ...
					+ mod((0 : w_width-1) - line_l(row) + first_col, w_width);
				a_cols = fliplr(round((cols + 2 * floor(4 * m_x) + 1 ...
					- abs(cols - 2 * floor(4 * m_x) - 1)) / 2));
				new_row = 1 + mod(row - first_row, w_len_corner);
				Xk(row, a_cols) = Xk(row, a_cols) + data(new_row, end:-1:1);
			end
			
			Xk = rot90(Xk);
		end
		
		Xk = Xk .* lowpass;
		id_1 = (-floor(2 * m):floor(2 * m)) + floor(4 * m) + 1;
		id_2 = (-floor(2 * n):floor(2 * n)) + floor(4 * n) + 1;
		Xk(id_1, id_2) = Xk(id_1, id_2) .* highpass_next;
		id_1 = X_ul(1) + (0 : 2*floor(4*m));
		id_2 = X_ul(2) + (0 : 2*floor(4*n));
		X(id_1, id_2) = X(id_1, id_2) + Xk;
		X_ul = X_ul + floor(4 * [m, n]) - floor(2 * [m, n]);
		lowpass = lowpass_next;
	end
	
	if is_real, X = conj(X) + rot90(X, 2); end
	
	m = m / 2;
	n = n / 2;
	Xk = fftshift(fft2(ifftshift(Y{1}{1}))) / sqrt(numel(Y{1}{1}));
	id_1 = X_ul(1) + (0 : 2*floor(4*m));
	id_2 = X_ul(2) + (0 : 2*floor(4*n));
	X(id_1, id_2) = X(id_1, id_2) + Xk .* lowpass;
	
	m = M / 3;
	n = N / 3;
	Y = fftshift(fft2(ifftshift(Y{end}{1}))) / sqrt(numel(Y{end}{1}));
	X_ul = ceil(([M, N] + 1) / 2) - floor([m, n]);
	id_1 = X_ul(1) + (0 : 2*floor(m));
	id_2 = X_ul(2) + (0 : 2*floor(n));
	Y(id_1, id_2) = Y(id_1, id_2) .* highpass + X;
	X = fftshift(ifft2(ifftshift(Y))) * sqrt(numel(Y));
	if is_real, X = real(X); end
end

function [wl, wr] = fdct_window(X)
	[wl, wr] = deal(zeros(size(X)));
	X(abs(X) < 1e-16) = 0;
	wl((0 < X) & (X < 1)) ...
		= exp(1 - 1 ./ (1 - exp(1 - 1 ./ (1 - X((0 < X) & (X < 1))))));
	wl(X >= 1) = 1;
	wr((0 < X) & (X < 1)) ...
		= exp(1 - 1 ./ (1 - exp(1 - 1 ./ X((0 < X) & (X < 1)))));
	wr(X <= 0) = 1;
	C = sqrt(wl .^ 2 + wr .^ 2);
	wl = wl ./ C;
	wr = wr ./ C;
end
