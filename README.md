
# DROUGHT_INDICES_PYTHON/README.md
# This file provides a description of your package, installation, and usage.
# It will be displayed on your PyPI project page.
"""
# drought_indices_python

[![Python application](https://github.com/yourusername/drought_indices_python/actions/workflows/python-app.yml/badge.svg)](https://github.com/yourusername/drought_indices_python/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/drought_indices_python.svg)](https://pypi.org/project/drought_indices_python/)

A Python library for computing and analyzing various drought indices, designed for researchers and practitioners in climate and hydrology. This package aims to provide robust and scientifically verifiable implementations of key drought indicators.

**Currently under active development.**

## About the Author

**Your Name** (PhD Student, Texas A&M University)
Email: your.email@tamu.edu

## Features

The `drought_indices_python` package offers a growing suite of essential drought indices:

* **Potential Evapotranspiration (PET):**
    * `calculate_pet_thornthwaite`: Estimates PET using the widely recognized Thornthwaite (1948) method, based on temperature and daylight hours.
* **Standardized Precipitation Index (SPI):**
    * `calculate_spi`: Quantifies precipitation deficit or surplus over various timescales, utilizing a calibration period for robust statistical fitting.
* **Standardized Precipitation Evapotranspiration Index (SPEI):**
    * `calculate_spei`: A powerful drought indicator based on the difference between precipitation and potential evapotranspiration, aggregated and transformed to a standard normal variate using a defined calibration period.
* **Data Utilities:**
    * Functions for loading, preprocessing, and saving meteorological and hydrological data.

## Installation

You can install `drought_indices_python` directly from PyPI using pip:

```bash
pip install drought_indices_python
```

For development purposes, or to work with the latest unreleased features, you can install it locally in editable mode:

```bash
cd /path/to/your/DROUGHT_INDICES_PYTHON/project # Navigate to the outer DROUGHT_INDICES_PYTHON folder
pip install -e .
```

## Usage

Once installed, you can easily import and utilize the functions within your Python scripts:

```python
import numpy as np
from drought_indices_python import calculate_pet_thornthwaite, calculate_spi, calculate_spei, load_data

# Example: Calculate PET (assuming you have annual_heat_index_I)
# annual_I = 50.0 # This would come from your long-term temperature data
# pet_value = calculate_pet_thornthwaite(monthly_temperature=18.5, latitude=40.0, month=7, annual_heat_index_I=annual_I)
# print(f"Calculated PET: {pet_value:.2f} mm")

# Example: Calculate SPI (using dummy data for illustration)
# precip_data_example = np.random.rand(360) * 150 # 30 years of monthly data
# spi_values = calculate_spi(
#     precipitation_data=precip_data_example,
#     scale_months=6,
#     data_start_year=1990,
#     calibration_start_year=1990,
#     calibration_end_year=2019
# )
# print(f"First 5 SPI values (6-month): {spi_values[:5]}")

# Example: Calculate SPEI (using dummy data for illustration)
# pet_data_example = np.random.rand(360) * 80 # 30 years of monthly PET
# spei_values = calculate_spei(
#     precipitation_data=precip_data_example,
#     pet_data=pet_data_example,
#     scale_months=12,
#     data_start_year=1990,
#     calibration_start_year=1990,
#     calibration_end_year=2019
# )
# print(f"First 5 SPEI values (12-month): {spei_values[:5]}")

# Data Loading (placeholder for your implementation)
# data = load_data('path/to/your/climate_data.csv')
```

## Contributing

We welcome contributions to `drought_indices_python`! Whether you're fixing bugs, adding new features, or improving documentation, your help is greatly appreciated. Please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""
