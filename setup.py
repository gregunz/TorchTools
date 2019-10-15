#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='DL Toolbox in PyTorch',
      version='1.0.0',
      description='Deep Learning personal tools & implementations in PyTorch',
      author='Gregoire Clement',
      author_email='mail@gregunz.io',
      url='github.com/gregunz',
      packages=find_packages(),
      install_requires=[
          'torch==1.2',
          'torchvision==0.4.0',
          # 'pytorch-lightning==0.5.1',
      ])
