"""phastphase.data_generation package initializer.

Expose the submodule `data_generation_utils` so that importing
`from phastphase.data_generation import data_generation_utils`
works from notebooks and other modules.
"""
from . import data_generation_utils

__all__ = ["data_generation_utils"]
