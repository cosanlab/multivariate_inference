#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='multivariate_inference',
    version='0.1.0',
    description="Package to test and compare different multivariate inference methods.",
    long_description="",
    author="Eshin Jolly",
    author_email='eshin.jolly.gr@dartmouth.edu',
    url='https://github.com/ejolly/multivariate_inference',
    packages=find_packages(include=['multivariate_inference']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='multivariate_inference',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests'
)
