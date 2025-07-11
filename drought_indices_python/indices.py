
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

def calculate_spi(precipitation_data, scale):
    """
    Calculates the Standardized Precipitation Index (SPI) for a given scale.

    The SPI quantifies precipitation deficit or surplus over various timescales.
    It involves fitting a probability distribution (typically Gamma) to aggregated
    precipitation data and then transforming it into a standard normal distribution.

    Args:
        precipitation_data (array-like): A 1D array-like (list, numpy array, pandas Series)
                                         of historical monthly precipitation values (e.g., in mm).
                                         This data should ideally span many years (e.g., 30+ years)
                                         for robust distribution fitting.
        scale (int): The aggregation period in months (e.g., 1, 3, 6, 12, 24).
                     Must be a positive integer.

    Returns:
        numpy.ndarray: An array of SPI values corresponding to the input
                       precipitation data, after aggregation and transformation.
                       The length of the output array will be `len(precipitation_data) - scale + 1`.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If `precipitation_data` is empty, `scale` is invalid,
                    or if distribution fitting fails due to insufficient data.

    Example:
        >>> # Example: 12 months of precipitation data
        >>> precip = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] * 30 # 30 years of data
        >>> spi_3_month = calculate_spi(precip, scale=3)
        >>> print(f"First 5 SPI (3-month) values: {spi_3_month[:5]}")
    """
    if not isinstance(precipitation_data, (list, np.ndarray)):
        raise TypeError("precipitation_data must be a list or numpy array.")
    if len(precipitation_data) == 0:
        raise ValueError("precipitation_data cannot be empty.")
    if not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be a positive integer.")
    if len(precipitation_data) < scale:
        raise ValueError(f"precipitation_data length ({len(precipitation_data)}) must be at least equal to scale ({scale}).")

    precip_np = np.array(precipitation_data, dtype=float)

    # Step 1: Aggregate precipitation data over the specified scale
    # Use a rolling sum to get aggregated precipitation for each period
    aggregated_precip = np.convolve(precip_np, np.ones(scale), mode='valid')

    # Step 2: Fit a Gamma distribution to the aggregated precipitation data
    # Filter out zeros for Gamma distribution fitting, as Gamma is defined for x > 0
    # A common approach for SPI is to handle zeros separately or use a mixed distribution.
    # For simplicity, we'll fit to non-zero values and then adjust.
    non_zero_aggregated_precip = aggregated_precip[aggregated_precip > 0]

    if len(non_zero_aggregated_precip) < 2: # Need at least 2 points to fit a distribution
        raise ValueError("Insufficient non-zero aggregated precipitation data to fit a distribution.")

    # Fit Gamma distribution (shape, loc, scale)
    # loc is often fixed at 0 for precipitation data
    shape, loc, scale_param = stats.gamma.fit(non_zero_aggregated_precip, floc=0)

    # Step 3: Transform aggregated precipitation to standard normal distribution
    spi_values = np.zeros_like(aggregated_precip, dtype=float)

    for i, val in enumerate(aggregated_precip):
        if val == 0:
            # Handle zero precipitation: assign a very small non-zero value or a specific SPI value
            # For simplicity in this example, we'll assign a very low SPI value,
            # or you could assign a fixed large negative value for extreme drought.
            # A more rigorous approach involves mixed distributions or conditional probabilities.
            spi_values[i] = stats.norm.ppf(stats.gamma.cdf(1e-6, shape, loc, scale_param)) # Map tiny value to SPI
        else:
            # Calculate the cumulative probability (CDF) for the aggregated value
            cdf_value = stats.gamma.cdf(val, shape, loc, scale_param)
            # Transform the CDF value to the corresponding Z-score (standard normal deviate)
            spi_values[i] = stats.norm.ppf(cdf_value)
            
    return spi_values

def calculate_pdsi(temperature, precipitation, awc, initial_pdsi=0):
    """
    Calculates the Palmer Drought Severity Index (PDSI).
    This is a placeholder function.

    Args:
        temperature (array-like): Time series of temperature data.
        precipitation (array-like): Time series of precipitation data.
        awc (float): Available Water Capacity of the soil.
        initial_pdsi (float): Initial PDSI value for the first period.

    Returns:
        array-like: Calculated PDSI values.
    """
    # Placeholder for PDSI calculation logic
    pass
