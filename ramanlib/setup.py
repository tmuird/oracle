"""Setup script for ramanlib."""

from setuptools import setup, find_packages

setup(
    name="ramanlib",
    version="0.1.0",
    description="Library for Raman spectroscopy analysis and fluorescence decomposition",
    author="Tom",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "torch>=2.0.0",
        "xarray>=2023.0.0",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
