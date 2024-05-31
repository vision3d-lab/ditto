from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension


numpy_include_dir = np.get_include()

triangle_hash_module1 = Extension(
    "lib.libmesh.triangle_hash",
    sources=["lib/libmesh/triangle_hash.pyx"],
    libraries=["m"],  # Unix-like specific
    include_dirs=[numpy_include_dir],
)


ext_modules = [
    triangle_hash_module1,
]

setup(ext_modules=cythonize(ext_modules), cmdclass={"build_ext": BuildExtension})
