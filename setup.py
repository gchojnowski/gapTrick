#!/usr/bin/env python

from setuptools import setup, find_packages

import subprocess
import os,sys,re
sys.path.insert(0, f'{os.path.dirname(__file__)}/gapTrick')
import gapTrick

def get_git_describe():
    import gapTrick.version
    return gapTrick.version.__version__

VERSION = get_git_describe()

setup(name='gapTrick',
      version=VERSION,
      description='rebuilds and completes models of protein complexes using AlphaFold2',
      url='https://github.com/gchojnowski/gapTrick',
      author='Grzegorz Chojnowski',
      author_email='gchojnowski@embl-hamburg.de',
      license='BSD',
      packages=['gapTrick'],
      install_requires=['af2plots', 'matplotlib'],
      entry_points={
          "console_scripts": [
            "gapTrick = gapTrick.__main__:main",
            ],
      }
     )