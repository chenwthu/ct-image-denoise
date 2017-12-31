function J = modified_curvelet_denoise(I, sigma)
	C = fdct(fftshift(ifft2(ones(size(I)))) * sqrt(numel(I)));
	
	E = cell(size(C));
	for u = 1:length(C)
		E{u} = cell(size(C{u}));
		for v = 1:length(C{u})
			A = C{u}{v};
			E{u}{v} = sqrt(sum(sum(A .* conj(A))) / numel(A));
		end
	end
	
	[F_rows, F_cols, n_rows, n_cols] = loc(C, size(I, 1), size(I, 2));
	C = fdct(circshift(I, [1, 1]));
	for u = 1:length(C)
		T = (3 + (u == length(C))) * sigma;
		for v = 1:length(C{u})
			if mod(ceil(v * 4 / length(C{u})), 2) == 1
				shift = F_cols{u}{v} * size(I, 2) / n_cols{u}{v};
				if v > 1, shift = shift / F_rows{u}{v}; end
				shift = -round(shift);
				value = sqrt(C{u}{v} .^ 2 + (circshift(abs(C{u}{v}), [0, 1]) .^ 2 ...
					+ circshift(abs(C{u}{v}), [0, -1]) .^ 2 ...
					+ circshift(abs(C{u}{v}), [1, shift]) .^ 2 ...
					+ circshift(abs(C{u}{v}), -[1, shift]) .^ 2) / 2);
			else
				shift = F_rows{u}{v} * size(I, 1) / n_rows{u}{v};
				if v > 1, shift = shift / F_cols{u}{v}; end
				shift = -round(shift);
				value = sqrt(C{u}{v} .^ 2 + (circshift(abs(C{u}{v}), [1, 0]) .^ 2 ...
					+ circshift(abs(C{u}{v}), [-1, 0]) .^ 2 ...
					+ circshift(abs(C{u}{v}), [shift, 1]) .^ 2 ...
					+ circshift(abs(C{u}{v}), -[shift, 1]) .^ 2) / 2);
			end
			C{u}{v} = C{u}{v} ./ abs(C{u}{v}) ...
				.* (abs(C{u}{v}) .* (value > T * E{u}{v}));
		end
	end
	
	J = circshift(real(ifdct(C, size(I, 1), size(I, 2))), -[1, 1]);
	J = J - min(J(:));
	J = J / max(J(:));
end

function Y = fdct(X)
	X = fftshift(fft2(ifftshift(X))) / sqrt(numel(X));
	[M, N] = size(X);
	n_scales = ceil(log2(min(M, N)) - 3);
	n_angles = [1, 8 * 2 .^ ceil((0 : n_scales-2) / 2)];
	Y = cell(1, n_scales);
	for k = 1:n_scales, Y{k} = cell(1, n_angles(k)); end
	
	m = M / 3;
	n = N / 3;
	id_1 = 1 + mod(floor(M / 2) - floor(2 * m) + (0 : 2*floor(2*m)), M);
	id_2 = 1 + mod(floor(N / 2) - floor(2 * n) + (0 : 2*floor(2*n)), N);
	X = X(id_1, id_2);
	w_len = floor(2 * [m, n]) - floor([m, n]) - 1 - (mod([M, N], 3) == 0);
	[wl_1, wr_1] = fdct_window(0 : 1/w_len(1) : 1);
	[wl_2, wr_2] = fdct_window(0 : 1/w_len(2) : 1);
	lowpass_1 = [wl_1, ones(1, 2 * floor(m) + 1), wr_1];
	if mod(M, 3) == 0, lowpass_1 = [0, lowpass_1, 0]; end
	lowpass_2 = [wl_2, ones(1, 2 * floor(n) + 1), wr_2];
	if mod(N, 3) == 0, lowpass_2 = [0, lowpass_2, 0]; end
	lowpass = lowpass_1' * lowpass_2;
	X_low = X .* lowpass;
	
	for k = n_scales:-1:2
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
		
		n_quad_angles = n_angles(k) / 4;
		l = 0;
		for q = 1:4
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
			Y{k}{l} = fftshift(ifft2(ifftshift(data))) * sqrt(numel(data));
			
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
				Y{k}{l} = fftshift(ifft2(ifftshift(data))) * sqrt(numel(data));
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
			Y{k}{l} = fftshift(ifft2(ifftshift(data))) * sqrt(numel(data));
			
			if q < 4, X_high = rot90(X_high); end
		end
	end
	
	Y{1}{1} = fftshift(ifft2(ifftshift(X_low))) * sqrt(numel(X_low));
end

function X = ifdct(Y, M, N)
	n_scales = length(Y);
	n_angles = [1, length(Y{2}) .* 2 .^ ceil((0 : n_scales-2) / 2)];
	
	m = M / 3;
	n = N / 3;
	X = zeros(2 * floor(2 * m) + 1, 2 * floor(2 * n) + 1);
	w_len = floor(2 * [m, n]) - floor([m, n]) - 1 - (mod([M, N], 3) == 0);
	[wl_1, wr_1] = fdct_window(0 : 1/w_len(1) : 1);
	[wl_2, wr_2] = fdct_window(0 : 1/w_len(2) : 1);
	lowpass_1 = [wl_1, ones(1, 2 * floor(m) + 1), wr_1];
	if mod(M, 3) == 0, lowpass_1 = [0, lowpass_1, 0]; end
	lowpass_2 = [wl_2, ones(1, 2 * floor(n) + 1), wr_2];
	if mod(N, 3) == 0, lowpass_2 = [0, lowpass_2, 0]; end
	lowpass = lowpass_1' * lowpass_2;
	
	X_ul = [1, 1];
	for k = n_scales:-1:2
		m = m / 2;
		n = n / 2;
		w_len = floor(2 * [m, n]) - floor([m, n]) - 1;
		[wl_1, wr_1] = fdct_window(0 : 1/w_len(1) : 1);
		[wl_2, wr_2] = fdct_window(0 : 1/w_len(2) : 1);
		lowpass_next = [wl_1, ones(1, 2 * floor(m) + 1), wr_1]' ...
			* [wl_2, ones(1, 2 * floor(n) + 1), wr_2];
		highpass_next = sqrt(1 - lowpass_next .^ 2);
		Xk = zeros(2 * floor(4 * m) + 1, 2 * floor(4 * n) + 1);
		
		n_quad_angles = n_angles(k) / 4;
		l = 0;
		for q = 1:4
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
			data = fftshift(fft2(ifftshift(Y{k}{l}))) / sqrt(numel(Y{k}{l}));
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
				data = fftshift(fft2(ifftshift(Y{k}{l}))) / sqrt(numel(Y{k}{l}));
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
			data = fftshift(fft2(ifftshift(Y{k}{l}))) / sqrt(numel(Y{k}{l}));
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
	
	m = m / 2;
	n = n / 2;
	Xk = fftshift(fft2(ifftshift(Y{1}{1}))) / sqrt(numel(Y{1}{1}));
	id_1 = X_ul(1) + (0 : 2*floor(4*m));
	id_2 = X_ul(2) + (0 : 2*floor(4*n));
	X(id_1, id_2) = X(id_1, id_2) + Xk .* lowpass;
	
	m = M / 3;
	n = N / 3;
	s = floor(2 * n) - floor(N / 2);
	Y = X(:, s+1 : s+N);
	Y(:, N-s+1 : N) = Y(:, N-s+1 : N) + X(:, 1:s);
	Y(:, 1:s) = Y(:, 1:s) + X(:, N+s+1 : N+s+s);
	s = floor(2 * m) - floor(M / 2);
	X = Y(s+1 : s+M, :);
	X(M-s+1 : M, :) = X(M-s+1 : M, :) + Y(1:s, :);
	X(1:s, :) = X(1:s, :) + Y(M+s+1 : M+s+s, :);
	X = fftshift(ifft2(ifftshift(X))) * sqrt(numel(X));
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

function [F_rows, F_cols, n_rows, n_cols] = loc(C, M, N)
	n_scales = length(C);
	n_angles = [1, length(C{2}) .* 2 .^ ceil((0 : n_scales-2) / 2)];
	
	m = M / 3;
	n = N / 3;
	[F_rows, F_cols, n_rows, n_cols] = deal(cell(1, n_scales));
	for k = 1:n_scales
		[F_rows{k}, F_cols{k}, n_rows{k}, n_cols{k}] = deal(cell(1, n_angles(k)));
	end
	
	dt = floor([M, N] / 2) - floor(2 * [m, n]);
	for k = n_scales:-1:2
		m = m / 2;
		n = n / 2;
		dt = dt + floor(4 * [m, n]) - floor(2 * [m, n]);
		
		n_quad_angles = n_angles(k) / 4;
		l = 0;
		for q = 1:4
			m_x = m * (mod(q, 2) == 0) + n * (mod(q, 2) == 1);
			m_y = m * (mod(q, 2) == 1) + n * (mod(q, 2) == 0);
			wl = round((0 : .5/n_quad_angles : .5) * 2 * floor(4 * m_x) + 1);
			wr = 2 * floor(4 * m_x) - wl + 2;
			w = [wl, wr(end+mod(n_quad_angles,2)-1 : -1 : 1)];
			w_end = w(2 : 2 : end-1);
			
			l = l + 1;
			first_w_end_y = round(floor(4 * m_y) / n_quad_angles + 1);
			w_len_corner = floor(4 * m_y) - floor(m_y) + ceil(first_w_end_y / 4);
			w_width = w_end(1) + w_end(2) - 1;
			w_slope = (floor(4 * m_x) + 1 - w_end(1)) / floor(4 * m_y);
			if q == 1
				w1 = dt(1);
				w2 = floor(N / 2) + 1 + w_slope * (w1 - floor(M / 2) - 1);
			elseif q == 2
				w2 = N + 1 - dt(2);
				w1 = floor(M / 2) + 1 - w_slope * (w2 - floor(N / 2) - 1);
			elseif q == 3
				w1 = M + 1 - dt(1);
				w2 = floor(N / 2) + 1 + w_slope * (w1 - floor(M / 2) - 1);
			else
				w2 = dt(2);
				w1 = floor(M / 2) + 1 - w_slope * (w2 - floor(N / 2) - 1);
			end
			F_rows{k}{l} = w1 - ceil((M + 1) / 2);
			F_cols{k}{l} = w2 - ceil((N + 1) / 2);
			n_rows{k}{l} = w_width * (mod(q, 2) == 0) ...
				+ w_len_corner * (mod(q, 2) == 1);
			n_cols{k}{l} = w_width * (mod(q, 2) == 1) ...
				+ w_len_corner * (mod(q, 2) == 0);
			
			w_len = floor(4 * m_y) - floor(m_y);
			for subl = 2 : n_quad_angles-1
				l = l + 1;
				w_width = w_end(subl + 1) - w_end(subl - 1) + 1;
				w_slope = (floor(4 * m_x) + 1 - w_end(subl)) / floor(4 * m_y);
				if q == 1
					w1 = dt(1);
					w2 = floor(N / 2) + 1 + w_slope * (w1 - floor(M / 2) - 1);
				elseif q == 2
					w2 = N + 1 - dt(2);
					w1 = floor(M / 2) + 1 - w_slope * (w2 - floor(N / 2) - 1);
				elseif q == 3
					w1 = M + 1 - dt(1);
					w2 = floor(N / 2) + 1 + w_slope * (w1 - floor(M / 2) - 1);
				else
					w2 = dt(2);
					w1 = floor(M / 2) + 1 - w_slope * (w2 - floor(N / 2) - 1);
				end
				F_rows{k}{l} = w1 - ceil((M + 1) / 2);
				F_cols{k}{l} = w2 - ceil((N + 1) / 2);
				n_rows{k}{l} = w_width * (mod(q, 2) == 0) ...
					+ w_len * (mod(q, 2) == 1);
				n_cols{k}{l} = w_width * (mod(q, 2) == 1) ...
					+ w_len * (mod(q, 2) == 0);
			end
			
			l = l + 1;
			w_width = 4 * floor(4 * m_x) + 3 - w_end(end) - w_end(end - 1);
			w_slope = (floor(4 * m_x) + 1 - w_end(end)) / floor(4 * m_y);
			if q == 1
				w1 = dt(1);
				w2 = floor(N / 2) + 1 + w_slope * (w1 - floor(M / 2) - 1);
			elseif q == 2
				w2 = N + 1 - dt(2);
				w1 = floor(M / 2) + 1 - w_slope * (w2 - floor(N / 2) - 1);
			elseif q == 3
				w1 = M + 1 - dt(1);
				w2 = floor(N / 2) + 1 + w_slope * (w1 - floor(M / 2) - 1);
			else
				w2 = dt(2);
				w1 = floor(M / 2) + 1 - w_slope * (w2 - floor(N / 2) - 1);
			end
			F_rows{k}{l} = w1 - ceil((M + 1) / 2);
			F_cols{k}{l} = w2 - ceil((N + 1) / 2);
			n_rows{k}{l} = w_width * (mod(q, 2) == 0) ...
				+ w_len_corner * (mod(q, 2) == 1);
			n_cols{k}{l} = w_width * (mod(q, 2) == 1) ...
				+ w_len_corner * (mod(q, 2) == 0);
		end
	end
	
	[F_rows{1}{1}, F_cols{1}{1}] = deal(0);
	[n_rows{1}{1}, n_cols{1}{1}] = size(C{1}{1});
end
