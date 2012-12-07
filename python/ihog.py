"""
This file implements the paired dictionary inference algorithm for HOG inversion
with Python. For learning the dictionary, see the MATLAB port.
"""

from numpy import *
import spams

class PairedDictionary(object):
    def __init__(self, dgray, dhog, n, k, ny, nx, lambda1, sbin):
        self.dgray = dgray
        self.dhog = dhog
        self.n = n
        self.k = k
        self.ny = ny
        self.nx = nx
        self.lambda1 = lambda1
        self.sbin = sbin

def invertHOG(feat, pd = None):
    ny = feat.shape[0]
    nx = feat.shape[1]

    windows = zeros((pd.ny*pd.nx*32, (ny-pd.ny+1)*(nx-pd.nx+1)))
    c = 0;
    for i in range(feat.shape[0] - pd.ny):
        for j in range(feat.shape[1] - pd.nx):
            hog = feat[i:pd.ny, j:pd.nx, :].flatten()
            hog = hog - hog.mean()
            hog = hog / sqrt(hog.variance() + 1)
            windows[:, c] = hog
            c += 1

    alpha = spams.lasso(windows, dhog, lambda1 = pd.lambda1, mode = 2)

    recon = dot(pd.dgray, alpha)

if __name__ == "__main__":
    pass
