from setuptools import setup

setup(
    name="torkit",
    version="0.1.0.rc0",
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
    url="https://github.com/Jiayuan-Gu/torkit",
    packages=["torkit"],
    long_description="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
