"""
setup.py - this module makes the package installable
"""

from setuptools import setup

NAME = "phastphase"
VERSION = "0.0.82"
DEPENDENCIES = ["numpy", "pytorch-minimize", "jax"]
DESCRIPTION = (
    "Accurate solution to the phase retrieval problem for near-Schwarz objects."
)
AUTHOR = "phastphase developers"
AUTHOR_EMAIL = "cbrabec@mit.edu"

setup(
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    install_requires=DEPENDENCIES,
    name=NAME,
    version=VERSION,
)
