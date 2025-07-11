
# DROUGHT_INDICES_PYTHON/README.md
# This file provides a description of your package, installation, and usage.
# It will be displayed on your PyPI project page.
"""
# drought_indices_python

A Python package designed for calculating and analyzing various drought indices, and handling related meteorological data. This project is currently in its early development phase, providing a skeleton for future implementation.

## Features (Planned)

* **Drought Index Calculation:**
    * Potential Evapototranspiration (PET)
    * Standardized Precipitation Index (SPI)
    * Palmer Drought Severity Index (PDSI)
    * And more!
* **Data Utilities:**
    * Loading and preprocessing meteorological data.
    * Saving results.

## Installation (Once deployed to PyPI)

You will be able to install `drought_indices_python` using pip:

```bash
pip install drought_indices_python
```

For development purposes, you can install it locally in editable mode:

```bash
cd /path/to/your/DROUGHT_INDICES_PYTHON/project # Navigate to the outer DROUGHT_INDICES_PYTHON folder
pip install -e .
```

## Usage (Planned)

Once the functions are implemented, you'll be able to use `drought_indices_python` like this:

```python
from drought_indices_python import calculate_pet, calculate_spi, load_data

# Example: Load some data (placeholder)
# data = load_data('path/to/your/data.csv')

# Example: Calculate PET (placeholder)
# pet_values = calculate_pet(temperature=data['temp'], latitude=30.0, month=7)

# Example: Calculate SPI (placeholder)
# spi_values = calculate_spi(precipitation_data=data['precip'], scale=6)

# print("PET values:", pet_values)
# print("SPI values:", spi_values)
```

## Contributing

We welcome contributions to `drought_indices_python`! If you'd like to help fill in the functions or add new features, please fork the repository, make your changes, and submit a pull request.
