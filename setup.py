#!/usr/bin/env python
import os
from setuptools import setup, find_packages

REQUIREMENTS = ['torch', 'numpy', 'scipy', 'pandas', 'hotelling', 'bootstrapped', 'openimages']

setup(name='whats_your_bench',
      version=0.1,
      description='Benchmarking suite for probabilistic programming languages',
      author='Patrick Pagni',
      author_email='patrick.pagni1@gmail.com',
      url='https://github.com/patrick-pagni/whats_your_bench',
      long_description=open('README.md').read(),
      packages=find_packages(),
      install_requires=REQUIREMENTS,
     )