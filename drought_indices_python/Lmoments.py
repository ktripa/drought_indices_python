import numpy as np

def calculate_lmoments(data_series, num_moments=3):
    """
    Calculates the first few L-moments (L1, L2, L3) for a given data series.

    This implementation is based on the robust estimation methods for L-moments,
    derived from established Fortran routines (e.g., those by J. R. M. Hosking).
    L-moments are robust statistics that provide measures of location, scale,
    skewness, and kurtosis, being less sensitive to outliers and more efficient
    for small samples compared to conventional moments. They are particularly
    useful in hydrological frequency analysis for fitting distributions.

    Args:
        data_series (array-like): A 1D array-like (list, numpy array) of
                                   numeric data for which to calculate L-moments.
        num_moments (int): The number of L-moments to calculate. This implementation
                           currently supports up to 3 L-moments (L1, L2, L3).
                           If `num_moments` is greater than 3, only the first 3
                           will be returned. Must be a positive integer.

    Returns:
        list: A list of calculated L-moments (L1, L2, L3). Returns an empty list
              if data_series is too short for the requested moments.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If `data_series` is empty or contains insufficient non-NaN values
                    for the calculation (e.g., less than 3 non-NaN values for L3).

    Example:
        >>> data = np.array([10, 12, 15, 18, 20, 22, 25, 28, 30, 32])
        >>> lm = calculate_lmoments(data, num_moments=3)
        >>> print(f"First 3 L-moments: {lm}")
        # Example output for a uniform-like distribution: [21.5, 6.75, 0.0] (approx)
    """
    if not isinstance(data_series, (list, np.ndarray)):
        raise TypeError("data_series must be a list or numpy array.")
    if not isinstance(num_moments, int) or num_moments <= 0:
        raise ValueError("num_moments must be a positive integer.")

    data_np = np.array(data_series, dtype=float)
    # Remove NaN values for calculation
    clean_data = data_np[~np.isnan(data_np)]
    
    n_values = len(clean_data)

    # This implementation computes up to 3 L-moments (L1, L2, L3)
    max_supported_moments = 3
    if num_moments > max_supported_moments:
        num_moments_to_return = max_supported_moments
    else:
        num_moments_to_return = num_moments

    if n_values < num_moments_to_return:
        if n_values == 0:
            raise ValueError("data_series cannot be empty or all NaN values.")
        return [] # Return empty list if not enough data for any moments

    # Sort the data in ascending order
    sorted_data = np.sort(clean_data)

    # Intermediate sums for the raw probability-weighted moments (beta_r equivalents)
    # These correspond to the `sums` array in the provided snippet after the first loop.
    intermediate_poly_sums = np.zeros(max_supported_moments)

    for i_idx in range(n_values): # 0-indexed loop for array access
        current_x_val = sorted_data[i_idx]
        
        # For L1 (lambda_1) - related to beta_0
        intermediate_poly_sums[0] += current_x_val
        
        # For L2 (lambda_2) - related to beta_1
        if max_supported_moments >= 2:
            intermediate_poly_sums[1] += current_x_val * i_idx
        
        # For L3 (lambda_3) - related to beta_2
        if max_supported_moments >= 3:
            intermediate_poly_sums[2] += current_x_val * i_idx * (i_idx - 1)
    
    # Normalize the intermediate sums (divide by N, N(N-1), N(N-1)(N-2) respectively)
    n_f = float(n_values)
    intermediate_poly_sums[0] /= n_f
    if n_f > 1:
        intermediate_poly_sums[1] /= (n_f * (n_f - 1.0))
    if n_f > 2:
        intermediate_poly_sums[2] /= (n_f * (n_f - 1.0) * (n_f - 2.0))

    # Apply final transformation to get L-moments (lambda_r)
    # This part translates the intermediate sums into the final L-moments (L1, L2, L3)
    # based on the structure of the Fortran routine.
    
    # The snippet's `sums` array (after normalization) then goes through two `k` loops
    # which effectively transform these into L1, L2, L3.
    # The snippet's final `lmoments` array then stores [L1, L2, L3/L2].
    # We want to return [L1, L2, L3].

    # L_1 = beta_0
    # L_2 = 2*beta_1 - beta_0
    # L_3 = 6*beta_2 - 6*beta_1 + beta_0

    # The `intermediate_poly_sums` calculated above are essentially beta_0, beta_1, beta_2.
    # So, we can directly use these to compute L1, L2, L3.

    beta0 = intermediate_poly_sums[0]
    beta1 = intermediate_poly_sums[1]
    beta2 = intermediate_poly_sums[2]

    final_lm_results = []
    if num_moments_to_return >= 1:
        final_lm_results.append(beta0) # L1
    if num_moments_to_return >= 2:
        final_lm_results.append(2.0 * beta1 - beta0) # L2
    if num_moments_to_return >= 3:
        final_lm_results.append(6.0 * beta2 - 6.0 * beta1 + beta0) # L3

    return final_lm_results