# warpaffinemask.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from cython cimport boundscheck, wraparound
from cython.parallel import prange
import cython
cimport numpy as np

@boundscheck(False)
@wraparound(False)
def rescale_blocks(double[:, :] img not None,
                   double[:, :] out not None,
                   int block_size,
                   bint parallel=False):
    """
    img   -- input 2D array (dtype=float64)
    out   -- pre-allocated output array, same shape as img
    block_size -- size of square tile
    parallel    -- if True, outer loops use prange()
    """
    cdef int new_h = img.shape[0]
    cdef int new_w = img.shape[1]
    cdef int xw, yh, bh, bw, i, j
    cdef double mn, mx, v

    if parallel:
        # parallel outer loop
        outer = prange
    else:
        # normal Python range
        outer = range

    # tile over rows
    for xw in outer(0, new_h, block_size):
        # tile over cols
        for yh in range(0, new_w, block_size):
            # actual tile dims
            bh = block_size if new_h - xw >= block_size else new_h - xw
            bw = block_size if new_w - yh >= block_size else new_w - yh

            # find block min/max
            mn = img[xw, yh]
            mx = mn
            for i in range(bh):
                for j in range(bw):
                    v = img[xw+i, yh+j]
                    if v < mn:
                        mn = v
                    elif v > mx:
                        mx = v

            # rescale + round into output
            for i in range(bh):
                for j in range(bw):
                    v = img[xw+i, yh+j]
                    # formula: round((mx+1)*((v+1)-mn)/65535.0)
                    out[xw+i, yh+j] = round((mx + 1.0)*(v + 1.0 - mn)/65535.0)
