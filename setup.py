#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='beatmapml',
    version='0.1.0',
    description='Utilities for osu! beatmap in machine learning',
    author='Youmu Chan',
    author_email='johnmave126@gmail.com',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Games/Entertainment',
    ],
    url='https://github.com/johnmave126/beatmapml',
    install_requires=[
        'numpy',
        'opencv-python',
        'slider @ git+https://github.com/llllllllll/slider.git@master#egg=slider-0.1.0'
    ]
)
