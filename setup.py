#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
from setuptools import setup

if sys.hexversion < 0x02070000:
    raise RuntimeError("Python 2.7 or higher required")

setup(
    name="dstrf",
    version="0.0.3",
    packages=["dstrf"],
    description="dstrf = strf + dynamics",
    long_description="",
    install_requires=[
        "numpy>=1.10",
    ],
    author="Tyler Robbins",
    maintainer="C Daniel Meliza",
)
