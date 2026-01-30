from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
from pathlib import Path

# Compute repository root relative to this file (works regardless of cwd)
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

include_dirs = [
    str(REPO_ROOT / "include"),
    str(REPO_ROOT / "include" / "baha"),
]

ext_modules = [
    Pybind11Extension(
        "pybaha",
        [str(THIS_DIR / "pybaha.cpp")],
        include_dirs=include_dirs,
        cxx_std=17,
        extra_compile_args=["-O3"] if os.name != "nt" else ["/O2"],
    ),
]

setup(
    name="pybaha",
    version="1.0.0",
    author="Sethurathienam Iyer",
    author_email="",
    description="Python bindings for BAHA (Branch-Aware Holonomy Annealing)",
    long_description=(REPO_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/sethuiyer/baha",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
