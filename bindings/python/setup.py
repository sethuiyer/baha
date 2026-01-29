from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

# Identify include directory
include_dirs = [
    os.path.join(os.getcwd(), "include"),
    os.path.join(os.getcwd(), "include", "baha")
]

ext_modules = [
    Pybind11Extension(
        "pybaha",
        ["bindings/python/pybaha.cpp"],
        include_dirs=include_dirs,
        cxx_std=17,
        extra_compile_args=["-O3"] if os.name != "nt" else ["/O2"],
    ),
]

setup(
    name="pybaha",
    version="0.1.0",
    author="Sethurathienam Iyer",
    description="Python bindings for BAHA",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
