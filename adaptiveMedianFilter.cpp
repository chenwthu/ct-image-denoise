#include "mex.h"
#include "matrix.h"
#include <vector>
#include <algorithm>

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

    nlhs = 1;
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    double *J = (double *)mxGetPr(plhs[0]);

    std::vector<double> patch;
    int maxHSize = ((m < n) ? (m / 20) : (n / 20));
    if (maxHSize < 3) maxHSize = 3;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int hsize = 3; hsize <= maxHSize; hsize += 2) {
                patch.clear();
                for (int u = i - hsize / 2; u <= i + hsize / 2; ++u)
                    for (int v = j - hsize / 2; v <= j + hsize / 2; ++v)
                        patch.push_back(I[at(u, v, m, n)]);
                std::sort(patch.begin(), patch.end());
                double low = patch.front(), high = patch.back();
                double med = patch[patch.size() / 2];
                if ((low < med) && (med < high)) {
                    if ((low < I[at(i, j, m, n)]) && (I[at(i, j, m, n)] < high))
                        J[at(i, j, m, n)] = I[at(i, j, m, n)];
                    else
                        J[at(i, j, m, n)] = med;
                    break;
                }
                else J[at(i, j, m, n)] = I[at(i, j, m, n)];
            }
}

