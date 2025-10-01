# setup.py
#python setup.py build_ext --inplace

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="warp_affine_mask_rescale",
    ext_modules=cythonize("warpaffinemaskrescale.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)

