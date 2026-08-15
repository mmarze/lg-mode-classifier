"""
Dataset generation tools for Laguerre-Gaussian beam simulations.

This subpackage provides utilities for generating synthetic datasets
for machine learning, including beam generation, sensor simulation,
and dataset creation.
"""

from .sensor import Sensor

from .generate import generate_dataset

from .generate_efficiently import  generate_dataset as  generate_dataset_efficiently

__all__ = [
    "Sensor",
    "generate_dataset",
    "generate_dataset_efficiently"
]