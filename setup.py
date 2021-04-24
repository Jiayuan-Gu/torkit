from setuptools import setup, find_packages
import torkit

setup(
    name="torkit",
    version=torkit.__version__,
    author="Jiayuan Gu",
    author_email="jigu@eng.ucsd.edu",
    description="Pytorch Toolkit",
    install_requires=[
        'torch==1.5.1',
        'numpy',
        'scipy',
        'pytest',
        'yacs',
        'pyyaml',
        'loguru',
    ],
    python_requires='>=3.6',
    url="",
    packages=find_packages(include=['torkit'], exclude=("tests",)),
    long_description="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
