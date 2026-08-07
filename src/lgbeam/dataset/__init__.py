"""
Dataset generation tools for Laguerre-Gaussian beam simulations.

This subpackage provides utilities for generating synthetic datasets
for machine learning, including beam generation, sensor simulation,
and dataset creation.
"""

from .sensor import Sensor
# from .beam import generate_beam
# from .simulator import simulate
# from .generate import generate_dataset

__all__ = [
    "Sensor",
    # "generate_beam",
    # "simulate",
    # "generate_dataset",
]