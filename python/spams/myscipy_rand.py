import numpy as np
import scipy.sparse as ssp

def rand(m, n, density=0.01, format="coo", dtype=None):
    """Generate a sparse matrix of the given shape and density with uniformely
    distributed values.

    Parameters
    ----------
    m, n: int
        shape of the matrix
    density: real
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format: str
        sparse matrix format.
    dtype: dtype
        type of the returned matrix values.

    Notes
    -----
    Only float types are supported for now.
    """
    if density < 0 or density > 1:
        raise ValueError("density expected to be 0 <= density <= 1")
    if dtype and not dtype in [np.float32, np.float64, np.longdouble]:
        raise NotImplementedError("type %s not supported" % dtype)

    mn = m * n

    # XXX: sparse uses intc instead of intp...
    tp = np.intp
    if mn > np.iinfo(tp).max:
        msg = """\
Trying to generate a random sparse matrix such as the product of dimensions is
greater than %d - this is not supported on this machine
"""
        raise ValueError(msg % np.iinfo(tp).max)

    # Number of non zero values
    k = long(density * m * n)

    # Generate a few more values than k so that we can get unique values
    # afterwards.
    # XXX: one could be smarter here
    mlow = 5
    fac = 1.02
    gk = min(k + mlow, fac * k)

    def _gen_unique_rand(_gk):
        id = np.random.rand(_gk)
        return np.unique(np.floor(id * mn))[:k]

    id = _gen_unique_rand(gk)
    while id.size < k:
        gk *= 1.05
        id = _gen_unique_rand(gk)

    j = np.floor(id * 1. / m).astype(tp)
    i = (id - j * m).astype(tp)
    vals = np.random.rand(k).astype(dtype)
    return ssp.coo_matrix((vals, (i, j)), shape=(m, n)).asformat(format)

