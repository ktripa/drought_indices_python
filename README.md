
# drought_indices_python

A Python library for computing and analyzing various drought indices, designed for researchers and practitioners in climate and hydrology. This package aims to provide robust and scientifically verifiable implementations of key drought indicators.

**Currently under active development.**

## About the Author

**Kumar Puran Tripathy** (PhD Student, Texas A&M University)
Email: tripathypuranbdk@gmail.com

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