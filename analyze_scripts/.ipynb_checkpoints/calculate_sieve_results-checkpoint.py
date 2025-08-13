import pandas as pd

def calculate_values(sieve_diameter, passage_percentage, target_value, all_passing_percentage, all_sieve_diameter):
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
        # rate = (tv - pp) / sv
        rates.append(rate)

        # Calculate x
        x = (tv - pp) / rate
        x_values.append(x)

        # Calculate x + sieve_value
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


def get_results(name, target_value=[.10, .16, .20, .25, .30, .40, .50, .60, .70, .75, .80, .84, .90]):
    summary = '/projects/OLIVINE/data/sieve_results/Sieve Result Summary.csv'
    target_value = target_value

    tmp_df = pd.read_csv(summary)
    try:
        all_passing_percentage = list(tmp_df[name] / 100)
        all_sieve_diameter = list(tmp_df['Sieve Diameter'])
    except:
        print(f"{name} not found!")
        return
    # Empty lists for generated data.
    sieve_diameter = []
    passage_percentage = []
    # Generates passage_percentage and sieve diameter when given target value, all_passing_percentage, all_sieve_diameter
    for val in target_value:
        low_pp = None
        for percen in reversed(all_passing_percentage):
            if percen < val:
                if low_pp == None:
                    low_pp = percen
                else:
                    if percen > low_pp:
                        low_pp = percen
        passage_percentage.append(low_pp)
        tmp_index = all_passing_percentage.index(low_pp)
        sieve_diameter.append(all_sieve_diameter[tmp_index])

    # Calculate values and get the DataFrame.
    df = calculate_values(sieve_diameter, passage_percentage, target_value, all_passing_percentage, all_sieve_diameter)

    # Display the table.
    return df, tmp_df

# --------------------------------------- #
# Use case:
# --------------------------------------- #
for name in ["South_High", "North_Low", "North_High", "South_Low", "Virginia_Beach", "Venice_Beach", "OGT","Olivine", "Washington_State"]:
    sand_name = name
    result_df, _ = get_results(sand_name)
    result_df = result_df[['percentage passing', 'ground truth size']]
    result_df.to_csv(f'/projects/OLIVINE/data/sieve_results/{sand_name}.csv', index=False)