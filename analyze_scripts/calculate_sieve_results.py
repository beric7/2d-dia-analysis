# Copyright 2025 The MITRE Corporation

import pandas as pd
from typing import List, Tuple, Optional

def calculate_values(
    sieve_diameter: List[float],
    passage_percentage: List[float],
    target_value: List[float],
    all_passing_percentage: List[float],
    all_sieve_diameter: List[float]
) -> pd.DataFrame:
    """
    Calculate interpolation values for grain size analysis.

    Args:
        sieve_diameter (List[float]): Sieve diameters corresponding to the target values.
        passage_percentage (List[float]): Percentages passing for each sieve diameter.
        target_value (List[float]): Target percent passing values for interpolation.
        all_passing_percentage (List[float]): All available percent passing values from the dataset.
        all_sieve_diameter (List[float]): All available sieve diameters from the dataset.

    Returns:
        -> pd.DataFrame: DataFrame containing calculated rates, interpolated x values, and ground truth sizes.

    Output:
        DataFrame with columns:
            - passage_percentage
            - rate
            - percentage passing
            - start_value
            - sieve_value
            - x
            - x+sv
            - ground truth size
    """
    # Initialize lists to store calculated values
    rates = []
    x_values = []
    x_plus_sv_values = []

    # Iterate over each set of values
    for sv, pp, tv in zip(sieve_diameter, passage_percentage, target_value):
        # Calculate rate (m)
        pp_index = all_passing_percentage.index(pp)
        pp1 = all_passing_percentage[pp_index-1]
        pp2 = all_passing_percentage[pp_index]
        sv_index = all_sieve_diameter.index(sv)
        sv1 = all_sieve_diameter[sv_index-1]
        sv2 = all_sieve_diameter[sv_index]
        rate = (pp2 - pp1) / (sv2 - sv1)
        rates.append(rate)

        # Calculate x (interpolated diameter)
        x = (tv - pp) / rate
        x_values.append(x)

        # Calculate x + sieve_value (final interpolated size)
        x_plus_sv = x + sv
        x_plus_sv_values.append(x_plus_sv)

    # Create a DataFrame with the calculated values
    df = pd.DataFrame({
        'passage_percentage': passage_percentage,
        'rate': rates,
        'percentage passing': target_value,
        'start_value': passage_percentage,
        'sieve_value': sieve_diameter,
        'x': x_values,
        'x+sv': x_plus_sv_values,
        'ground truth size': x_plus_sv_values
    })

    return df

def get_results(
    name: str,
    target_value: Optional[List[float]] = None
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Retrieve and interpolate sieve results for a given sand sample.

    Args:
        name (str): Name of the sand sample (must match a column in the summary CSV).
        target_value (Optional[List[float]]): List of target percent passing values (default is standard percentiles).

    Returns:
        -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
            - pd.DataFrame: DataFrame with interpolated ground truth sizes for each target value.
            - pd.DataFrame: Raw DataFrame loaded from the summary CSV.

    Output:
        Saves no files directly. Returns DataFrames for further use.
    """
    summary = '/projects/OLIVINE/data/sieve_results/Sieve Result Summary.csv'
    if target_value is None:
        target_value = [.10, .16, .20, .25, .30, .40, .50, .60, .70, .75, .80, .84, .90]

    tmp_df = pd.read_csv(summary)
    try:
        all_passing_percentage = list(tmp_df[name] / 100)
        all_sieve_diameter = list(tmp_df['Sieve Diameter'])
    except Exception:
        print(f"{name} not found!")
        return None

    # Find the lower bound passage percentage and corresponding sieve diameter for each target value
    sieve_diameter = []
    passage_percentage = []
    for val in target_value:
        low_pp = None
        for percen in reversed(all_passing_percentage):
            if percen < val:
                if low_pp is None:
                    low_pp = percen
                else:
                    if percen > low_pp:
                        low_pp = percen
        passage_percentage.append(low_pp)
        tmp_index = all_passing_percentage.index(low_pp)
        sieve_diameter.append(all_sieve_diameter[tmp_index])

    # Calculate values and get the DataFrame.
    df = calculate_values(sieve_diameter, passage_percentage, target_value, all_passing_percentage, all_sieve_diameter)

    return df, tmp_df

# --------------------------------------- #
# Use case:
# --------------------------------------- #
if __name__ == "__main__":
    """
    Example use case: For each sand sample, calculate and save ground truth sieve results.

    Output:
        Saves a CSV file for each sand sample with columns:
            - percentage passing
            - ground truth size
    """
    for name in [
        "South_High", "North_Low", "North_High", "South_Low",
        "Virginia_Beach", "Venice_Beach", "OGT", "Olivine", "Washington_State"
    ]:
        sand_name = name
        result = get_results(sand_name)
        if result is not None:
            result_df, _ = result
            result_df = result_df[['percentage passing', 'ground truth size']]
            result_df.to_csv(f'/projects/OLIVINE/data/sieve_results/{sand_name}.csv', index=False)
