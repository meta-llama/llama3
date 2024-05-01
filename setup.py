# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from setuptools import find_packages, setup
from llama import __version__ as VERSION


def get_requirements(path: str):
    return [l.strip() for l in open(path)]


setup(
    name="llama3",
    version=VERSION,
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
