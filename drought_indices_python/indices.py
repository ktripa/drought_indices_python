
import math
import numpy as np
from scipy import stats

def calculate_pet_thornthwaite(monthly_temperature, latitude, month, annual_heat_index_I):
    """
    Calculates Potential Evapotranspiration (PET) using the Thornthwaite (1948) method.

    The Thornthwaite equation is an empirical method that estimates PET based on
    mean monthly air temperature and mean daily daylight hours. The method requires
    the annual heat index (I), which is the sum of 12 monthly heat indices (i)
    derived from mean monthly temperatures over a full year.

    Args:
        monthly_temperature (float): The mean temperature for the specific month (in Celsius).
                                     Must be a single numeric value.
        latitude (float): The latitude of the location (in degrees, -90 to 90).
        month (int): The month of the year (1 for January, 12 for December).
                     Must be an integer between 1 and 12.
        annual_heat_index_I (float): The annual heat index for the location,
                                     calculated as the sum of monthly heat indices (i)
                                     for all 12 months of the year. This value must
                                     be pre-calculated and provided.

    Returns:
        float: The calculated Potential Evapotranspiration (PET) for the given month in mm.

    Raises:
        ValueError: If input values are invalid (e.g., month out of range,
                    latitude out of range, non-positive annual heat index).
        TypeError: If input types are incorrect.

    Example:
        To calculate monthly PET for a specific month, you first need the annual heat index.
        Let's assume for a location, the annual heat index (I) is 50.0.
        For July (month=7) with a mean temperature of 18.5°C at latitude 40.0°N:

        >>> annual_I = 50.0
        >>> pet_july = calculate_pet_thornthwaite(18.5, 40.0, 7, annual_I)
        >>> print(f"PET for July: {pet_july:.2f} mm")
        # Expected output will vary based on exact calculations, but it would be a specific mm value.
    """
    # Input validation
    if not isinstance(monthly_temperature, (int, float)):
        raise TypeError("monthly_temperature must be a numeric value.")
    if not isinstance(latitude, (int, float)):
        raise TypeError("latitude must be a numeric value.")
    if not (-90 <= latitude <= 90):
        raise ValueError("latitude must be between -90 and 90 degrees.")
    if not isinstance(month, int) or not (1 <= month <= 12):
        raise ValueError("month must be an integer between 1 and 12.")
    if not isinstance(annual_heat_index_I, (int, float)) or annual_heat_index_I <= 0:
        raise ValueError("annual_heat_index_I must be a positive numeric value.")

    # Step 1: Calculate monthly heat index (i) for the given month
    # Note: This 'i' is for the *current* month's temperature, not the sum for 'I'.
    # Thornthwaite's original formula uses unadjusted PET based on Tm.
    # If Tm <= 0, the monthly heat index is 0.
    if monthly_temperature <= 0:
        monthly_heat_index_i = 0
    else:
        monthly_heat_index_i = (monthly_temperature / 5.0)**1.514

    # Step 2: Calculate the exponent 'a' based on the annual heat index 'I'
    # This 'a' is a function of the annual heat index, not the monthly one.
    a = (0.49239 + (0.01792 * annual_heat_index_I) -
         (0.0000771 * annual_heat_index_I**2) +
         (0.000000675 * annual_heat_index_I**3))

    # Step 3: Calculate unadjusted PET (for a 30-day month with 12 hours of daylight)
    # This formula is applied if monthly_temperature > 0
    if monthly_temperature <= 0:
        unadjusted_pet = 0.0
    else:
        unadjusted_pet = 16.0 * ((10.0 * monthly_temperature) / annual_heat_index_I)**a

    # Step 4: Calculate monthly mean daily daylight hours (N_hours)
    # This requires Julian day calculation based on month
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # Using 28 for Feb for simplicity, exact day count not critical for mean daylight
    mid_month_day = sum(days_in_month[:month]) + (days_in_month[month] / 2.0)

    # Convert latitude to radians
    lat_rad = math.radians(latitude)

    # Declination angle (delta)
    delta = 0.409 * math.sin(((2 * math.pi / 365) * mid_month_day) - 1.39)

    # Sunset hour angle (omega_s)
    # Handle edge cases for tan(phi) * tan(delta) that might lead to math domain error
    tan_lat = math.tan(lat_rad)
    tan_delta = math.tan(delta)
    arg_acos = -tan_lat * tan_delta

    # Clamp arg_acos to [-1, 1] to prevent math domain errors for arccos
    arg_acos = max(-1.0, min(1.0, arg_acos))
    
    omega_s = math.acos(arg_acos)

    # Monthly mean daily daylight hours (N_hours)
    n_hours = (24.0 / math.pi) * omega_s

    # Step 5: Calculate monthly correction factor (k_m)
    # This adjusts for actual day length and days in month
    # Assuming average days in month for the calculation, or specific days for leap year etc.
    # For simplicity, using standard days in month (not considering leap year for Feb here)
    actual_days_in_month = days_in_month[month] # Using 28 for Feb, can be adjusted for leap year outside if needed

    # The correction factor is (N_hours / 12) * (Actual Days in Month / 30)
    # However, Thornthwaite's tables directly give the adjustment.
    # A common simplification is to use (N_hours / 12) * (days_in_month / 30)
    # Let's use the standard adjustment factor from the textbook formulas:
    # It's the unadjusted PET * a correction factor based on daylight hours and days in month
    # The 'k' factor is often directly incorporated by multiplying the unadjusted PET
    # by the ratio of actual daylight hours to 12, and actual days in month to 30.

    # A more precise k_m is often found in tables, but for calculation:
    k_m = n_hours / 12.0 # Ratio of actual daylight hours to 12 hours (standard)
    
    # Final PET calculation for the month
    # Multiply the unadjusted PET by the correction factor based on day length and number of days
    # The unadjusted PET is for a 30-day month with 12 hours of daylight.
    # So, we adjust by (actual_days_in_month / 30) and (n_hours / 12)
    final_pet = unadjusted_pet * (actual_days_in_month / 30.0) * (n_hours / 12.0)

    return final_pet

def calculate_spi(precipitation_data, scale_months, data_start_year, calibration_start_year, calibration_end_year):
    """
    Calculates the Standardized Precipitation Index (SPI) for a given timescale.

    The SPI quantifies precipitation deficit or surplus. It involves fitting a
    probability distribution (typically Gamma) to aggregated precipitation data
    from a specified calibration period, and then transforming the entire dataset
    to a standard normal distribution using those fitted parameters.

    Args:
        precipitation_data (array-like): A 1D array-like (list, numpy array)
                                         of historical monthly precipitation values (e.g., in mm).
                                         This data should span many years (e.g., 30+ years)
                                         and cover the full data period.
        scale_months (int): The aggregation period in months (e.g., 1, 3, 6, 12, 24).
                            Must be a positive integer.
        data_start_year (int): The calendar year corresponding to the first data point
                               in `precipitation_data`.
        calibration_start_year (int): The initial year of the calibration period.
        calibration_end_year (int): The final year of the calibration period.

    Returns:
        numpy.ndarray: An array of SPI values corresponding to the input
                       precipitation data, after aggregation and transformation.
                       The array will have the same length as `precipitation_data`,
                       with initial values (for periods less than `scale_months`) as NaN.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If `precipitation_data` is empty, `scale_months` is invalid,
                    calibration period is invalid, or if distribution fitting
                    fails due to insufficient data.

    Example:
        >>> # Example: 30 years of monthly precipitation data (360 months)
        >>> precip_data = np.random.rand(360) * 150
        >>> spi_6_month = calculate_spi(precip_data, scale_months=6,
        ...                             data_start_year=1990,
        ...                             calibration_start_year=1990,
        ...                             calibration_end_year=2019)
        >>> print(f"First 10 SPI (6-month) values: {spi_6_month[:10]}")
    """
    if not isinstance(precipitation_data, (list, np.ndarray)):
        raise TypeError("precipitation_data must be a list or numpy array.")
    if len(precipitation_data) == 0:
        raise ValueError("precipitation_data cannot be empty.")
    if not isinstance(scale_months, int) or scale_months <= 0:
        raise ValueError("scale_months must be a positive integer.")
    if len(precipitation_data) < scale_months:
        raise ValueError(f"precipitation_data length ({len(precipitation_data)}) must be at least equal to scale_months ({scale_months}).")
    if not all(isinstance(y, int) for y in [data_start_year, calibration_start_year, calibration_end_year]):
        raise TypeError("Year arguments must be integers.")
    if not (calibration_start_year >= data_start_year and calibration_end_year >= calibration_start_year):
        raise ValueError("Calibration period years are invalid or outside data range.")

    precip_np = np.array(precipitation_data, dtype=float)

    # Step 1: Aggregate precipitation data over the specified scale
    # Use a rolling sum to get aggregated precipitation for each period
    aggregated_precip = np.convolve(precip_np, np.ones(scale_months), mode='valid')

    # Step 2: Determine indices for calibration period in the *aggregated* array
    # The aggregated array starts 'scale_months - 1' months after the original data's start.
    months_from_data_start_to_cal_start = (calibration_start_year - data_start_year) * 12
    months_from_data_start_to_cal_end = (calibration_end_year - data_start_year + 1) * 12 # +1 to include end year

    # Adjust these indices to match the aggregated data's timeline
    cal_agg_start_idx = max(0, months_from_data_start_to_cal_start - (scale_months - 1))
    cal_agg_end_idx = min(len(aggregated_precip), months_from_data_start_to_cal_end - (scale_months - 1))

    calibration_data_aggregated = aggregated_precip[cal_agg_start_idx:cal_agg_end_idx]

    # Filter out zeros for Gamma distribution fitting, as Gamma is defined for x > 0
    non_zero_calibration_data = calibration_data_aggregated[calibration_data_aggregated > 0]

    if len(non_zero_calibration_data) < 2: # Need at least 2 points to fit a distribution
        raise ValueError("Insufficient non-zero calibration data to fit a distribution for SPI.")

    # Step 3: Fit Gamma distribution to calibration data
    # floc=0 fixes the location parameter at 0, common for precipitation data
    shape, loc, scale_param = stats.gamma.fit(non_zero_calibration_data, floc=0)

    # Step 4: Transform *all* aggregated precipitation to standard normal distribution
    spi_values_aggregated = np.zeros_like(aggregated_precip, dtype=float)

    for i, val in enumerate(aggregated_precip):
        if val == 0:
            # Handle zero precipitation: map a tiny value to its CDF for transformation
            # A more rigorous approach might use a mixed distribution.
            cdf_value = stats.gamma.cdf(1e-6, shape, loc, scale_param)
            spi_values_aggregated[i] = stats.norm.ppf(cdf_value)
        else:
            cdf_value = stats.gamma.cdf(val, shape, loc, scale_param)
            spi_values_aggregated[i] = stats.norm.ppf(cdf_value)
            
    # Step 5: Pad the beginning with NaNs to match original input length
    # The aggregated array is shorter by (scale_months - 1) elements at the beginning.
    final_spi_output = np.full(len(precipitation_data), np.nan)
    final_spi_output[scale_months - 1:] = spi_values_aggregated

    return final_spi_output


def calculate_spei(precipitation_data, pet_data, scale_months,
                   data_start_year, calibration_start_year, calibration_end_year):
    """
    Calculates the Standardized Precipitation Evapotranspiration Index (SPEI).

    The SPEI is a drought index based on the difference between precipitation (P)
    and potential evapotranspiration (PET) over various timescales. This P-PET
    difference is aggregated, fitted to a probability distribution (typically
    Gamma or Log-logistic), and then transformed into a standard normal variate.

    Args:
        precipitation_data (array-like): A 1D array-like (list, numpy array)
                                         of historical monthly precipitation values (e.g., in mm).
                                         Must be of the same length as `pet_data`.
        pet_data (array-like): A 1D array-like (list, numpy array) of historical
                               monthly potential evapotranspiration values (e.g., in mm).
                               Must be of the same length as `precipitation_data`.
        scale_months (int): The aggregation period in months (e.g., 1, 3, 6, 12, 24).
                            Must be a positive integer.
        data_start_year (int): The calendar year corresponding to the first data point
                               in `precipitation_data` and `pet_data`.
        calibration_start_year (int): The initial year of the calibration period.
        calibration_end_year (int): The final year of the calibration period.

    Returns:
        numpy.ndarray: An array of SPEI values corresponding to the input
                       data, after aggregation and transformation. The array
                       will have the same length as input `precipitation_data`,
                       with initial values (for periods less than `scale_months`) as NaN.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If input arrays have incompatible lengths, are empty,
                    `scale_months` is invalid, calibration period is invalid,
                    or if distribution fitting fails due to insufficient data.

    Example:
        >>> # Example: 30 years of monthly P and PET data (360 months)
        >>> precip_example = np.random.rand(360) * 150
        >>> pet_example = np.random.rand(360) * 80
        >>> spei_12_month = calculate_spei(precip_example, pet_example, scale_months=12,
        ...                                data_start_year=1990,
        ...                                calibration_start_year=1990,
        ...                                calibration_end_year=2019)
        >>> print(f"First 10 SPEI (12-month) values: {spei_12_month[:10]}")
    """
    # Input validation
    if not isinstance(precipitation_data, (list, np.ndarray)) or not isinstance(pet_data, (list, np.ndarray)):
        raise TypeError("precipitation_data and pet_data must be lists or numpy arrays.")
    if len(precipitation_data) != len(pet_data):
        raise ValueError("precipitation_data and pet_data must have the same length.")
    if len(precipitation_data) == 0:
        raise ValueError("Input data arrays cannot be empty.")
    if not isinstance(scale_months, int) or scale_months <= 0:
        raise ValueError("scale_months must be a positive integer.")
    if len(precipitation_data) < scale_months:
        raise ValueError(f"Input data length ({len(precipitation_data)}) must be at least equal to scale_months ({scale_months}).")
    if not all(isinstance(y, int) for y in [data_start_year, calibration_start_year, calibration_end_year]):
        raise TypeError("Year arguments must be integers.")
    if not (calibration_start_year >= data_start_year and calibration_end_year >= calibration_start_year):
        raise ValueError("Calibration period years are invalid or outside data range.")

    precip_np = np.array(precipitation_data, dtype=float)
    pet_np = np.array(pet_data, dtype=float)

    # Step 1: Calculate the difference D = P - PET
    diff_data = precip_np - pet_np

    # Step 2: Add an offset to make all values positive for Gamma distribution fitting.
    # This is a common practice when using Gamma for SPEI, as Gamma is defined for x > 0.
    # The offset is typically large enough (e.g., 1000mm) to ensure positivity.
    offset_value = 1000.0
    diff_data_offset = diff_data + offset_value

    # Step 3: Aggregate the P-PET difference over the specified scale
    aggregated_diff_offset = np.convolve(diff_data_offset, np.ones(scale_months), mode='valid')

    # Step 4: Determine indices for calibration period in the *aggregated* array
    # The aggregated array starts 'scale_months - 1' months after the original data's start.
    months_from_data_start_to_cal_start = (calibration_start_year - data_start_year) * 12
    months_from_data_start_to_cal_end = (calibration_end_year - data_start_year + 1) * 12 # +1 to include end year

    # Adjust these indices to match the aggregated data's timeline
    cal_agg_start_idx = max(0, months_from_data_start_to_cal_start - (scale_months - 1))
    cal_agg_end_idx = min(len(aggregated_diff_offset), months_from_data_start_to_cal_end - (scale_months - 1))

    calibration_data_aggregated = aggregated_diff_offset[cal_agg_start_idx:cal_agg_end_idx]

    # Filter out zeros for Gamma distribution fitting
    non_zero_calibration_data = calibration_data_aggregated[calibration_data_aggregated > 0]

    if len(non_zero_calibration_data) < 2:
        raise ValueError("Insufficient non-zero calibration data to fit a distribution for SPEI.")

    # Step 5: Fit Gamma distribution to calibration data
    # floc=0 fixes the location parameter at 0
    shape, loc, scale_param = stats.gamma.fit(non_zero_calibration_data, floc=0)

    # Step 6: Transform *all* aggregated P-PET data to standard normal distribution
    spei_values_aggregated = np.zeros_like(aggregated_diff_offset, dtype=float)

    for i, val in enumerate(aggregated_diff_offset):
        if val <= 0: # Should ideally not be <=0 after offset, but as a safeguard
            # Map tiny value to its CDF for transformation
            cdf_value = stats.gamma.cdf(1e-9, shape, loc, scale_param)
            spei_values_aggregated[i] = stats.norm.ppf(cdf_value)
        else:
            cdf_value = stats.gamma.cdf(val, shape, loc, scale_param)
            spei_values_aggregated[i] = stats.norm.ppf(cdf_value)
            
    # Step 7: Pad the beginning with NaNs to match original input length
    # The aggregated array is shorter by (scale_months - 1) elements at the beginning.
    final_spei_output = np.full(len(precipitation_data), np.nan)
    final_spei_output[scale_months - 1:] = spei_values_aggregated

    return final_spei_output