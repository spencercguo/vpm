#!/usr/bin/env python

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext


from setuptools import setup
from setuptools import find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='vpm',
    version='0.0.1',
    description='Variational power method',
    long_description=readme(),
    url='http://github.com/spencercguo/vpm',
    author='Spencer Guo',
    author_email='scguo@uchicago.edu',
    license='MIT',
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
    ],
    project_urls={
        "Issue Tracker": "https://github.com/chatipat/extq/issues",
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "jax",
    ],

)
