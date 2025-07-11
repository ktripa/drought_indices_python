# This file marks the 'drought_indices_python' directory as a Python package.
# It also helps expose key functions and defines the package version.

# Importing functions from the 'indices' and 'metrics' modules
from .indices import calculate_pet, calculate_spi, calculate_pdsi
from .metrics import load_data, preprocess_data, save_results # Renamed from data_utils to metrics

# Define the package version
__version__ = "0.1.0"

# List what should be imported when a user does 'from drought_indices_python import *'
__all__ = [
    "calculate_pet",
    "calculate_spi",
    "calculate_pdsi",
    "load_data",
    "preprocess_data",
    "save_results"
]
