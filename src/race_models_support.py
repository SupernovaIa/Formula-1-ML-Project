



def get_weather_condition(compounds):
    """
    Determine the race condition based on the proportion of tire compounds.

    Parameters
    -----------
    - compounds (pd.Series): A pandas Series where keys represent tire types (e.g., 'WET', 'INTERMEDIATE', etc.), 
    and values represent the proportions of each tire type.

    Returns
    --------
    - (str): The classified race condition, which can be:
    - "Wet" if more than 90% of tires are wet or intermediate.
    - "Mixed" if the proportion of wet or intermediate tires is between 10% and 90%.
    - "Dry" if the proportion of wet or intermediate tires is 10% or less.
    """

    # Thresholds to determine race conditions
    wet_threshold = 0.9
    mixed_threshold = 0.1

    # Calculate the total sum of the Series
    total = compounds.sum()

    # Ensure the values are normalized
    if total != 1:
        compounds = compounds / total

    # Proportion of wet tires
    wet = compounds.get('WET', 0) + compounds.get('INTERMEDIATE', 0)

    # Classify conditions based on the thresholds
    if wet > wet_threshold:
        return "wet"
    
    elif wet > mixed_threshold:
        return "mixed"
    
    else:
        return "dry"