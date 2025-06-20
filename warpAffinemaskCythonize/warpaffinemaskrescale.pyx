# warpaffinemaskrescale.pyx
import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound, nonecheck, cdivision

ctypedef fused pixel_t:
    cnp.uint16_t
    cnp.float64_t

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
@cdivision(True)
def warp_affine_mask_rescale(
        pixel_t[:, :] img not None,    # now accepts uint16 or float64
        cnp.float64_t[:, :] out not None,
        int block_size):
    cdef int H = img.shape[0], W = img.shape[1]
    cdef int x0, y0, x, y, bh, bw
    cdef double mn, mx, v, scale

    for x0 in range(0, H, block_size):
        for y0 in range(0, W, block_size):
            bh = block_size if H - x0 >= block_size else H - x0
            bw = block_size if W - y0 >= block_size else W - y0

            # first pass: find min/max
            mn = <double>img[x0, y0]; mx = mn
            for x in range(bh):
                for y in range(bw):
                    v = <double>img[x0 + x, y0 + y]
                    if v < mn: mn = v
                    elif v > mx: mx = v

            scale = (mx + 1.0) / 65535.0

            # second pass: rescale + round
            for x in range(bh):
                for y in range(bw):
                    v = <double>img[x0 + x, y0 + y]
                    out[x0 + x, y0 + y] = round(scale * (v + 1.0 - mn))
