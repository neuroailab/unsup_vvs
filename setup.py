#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "tensorflow==1.15.0",
    "torch",
    "tqdm",
    "sklearn",
    "pymongo",
    "netcdf4",
]

setup(
    name='unsup_vvs',
    version='0.1.0',
    long_description=readme,
    packages=find_packages(exclude=['tests']),
    install_requires=requirements,
)
