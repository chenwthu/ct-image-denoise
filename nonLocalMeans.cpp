#include "mex.h"
#include "matrix.h"
#include <cmath>

inline mwIndex at(int i, int j, int m, int n) {
    if (i < 0) i = - i - 1;
    if (i >= m) i = m * 2 - i - 1;
    if (j < 0) j = - j - 1;
    if (j >= n) j = n * 2 - j - 1;
    return j * m + i;
}

void mexFunction(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
    double *I = (double *)mxGetPr(prhs[0]);
    mwSize m = mxGetM(prhs[0]), n = mxGetN(prhs[0]);
    int r = *(double *)mxGetPr(prhs[1]);
    int f = *(double *)mxGetPr(prhs[2]);
    double sigma = *(double *)mxGetPr(prhs[3]);
    double h = *(double *)mxGetPr(prhs[4]);

    nlhs = 1;
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    double *J = (double *)mxGetPr(plhs[0]);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            J[at(i, j, m, n)] = 0;
            double sum = 0;
            for (int u = i - r; u <= i + r; ++u)
                for (int v = j - r; v <= j + r; ++v) {
                    double d = 0;
                    for (int x = -f; x <= f; ++x)
                        for (int y = -f; y <= f; ++y)
                            d += pow(I[at(i + x, j + y, m, n)] - I[at(u + x, v + y, m, n)], 2);
                    d /= pow(2 * f + 1, 2);
                    double w = exp(-std::fmax(d * d - 2 * sigma * sigma, 0) / (h * h));
                    J[at(i, j, m, n)] += I[at(u, v, m, n)] * w;
                    sum += w;
                }
            J[at(i, j, m, n)] /= sum;
        }
}

