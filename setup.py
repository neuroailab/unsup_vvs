#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'dotmap',  # for cmc
]

setup(
    name='unsup-vvs',
    version='0.1.0',
    description="Unsupervised models for the ventral visual stream",
    long_description=readme,
    author="Chengxu Zhuang",
    author_email='chengxuz@stanford.edu',
    url='https://github.com/neuroailab/unsup_vvs',
    packages=find_packages(exclude=['prepare_datasets']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='unsup-vvs',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
)
