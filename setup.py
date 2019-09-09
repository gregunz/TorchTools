#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='DL PyTorch',
      version='1.0',
      description='Deep Learning personal implementations in PyTorch',
      author='Gregoire Clement',
      author_email='mail@gregunz.io',
      url='github.com/gregunz',
      packages=find_packages(),
      requires=[
          'torch',
      ])
