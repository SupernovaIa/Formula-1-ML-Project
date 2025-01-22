



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
    

def get_extra_info(session, info):
    """
    Extracts additional information from a FastF1 session object and updates the provided info dictionary with metrics such as weather conditions, yellow flags, red flags, and safety car deployments.

    Parameters
    -----------
    - session (fastf1.core.Session): The FastF1 session object containing lap and track status data. Must include a 'laps' DataFrame with a 'Compound' column and a 'track_status' DataFrame with a 'Message' column.
    - info (dict): A dictionary where the keys 'weather', 'yellows', 'reds', 'sc', and 'vsc' must be present. Each key should map to a list that will be updated with the corresponding session data.

    Returns
    --------
    - None: Updates the `info` dictionary in place.
    """

    # Check keys
    if not all(key in info for key in ['weather', 'yellows', 'reds', 'sc', 'vsc']):
        raise ValueError("The 'info' dictionary must contain the following keys: 'weather', 'yellows', 'reds', 'sc', 'vsc'")
    
    # Check columns exists
    if 'Compound' not in session.laps.columns or 'Message' not in session.track_status.columns:
        raise KeyError("The columns 'Compound' or 'Message' are not available in the session data.")
    
    # Get compounds and track status
    compounds = session.laps['Compound'].value_counts(normalize=True)
    track_status_counts = session.track_status['Message'].value_counts()

    # Get metrics
    info['weather'].append(get_weather_condition(compounds))
    info['yellows'].append(track_status_counts.get('Yellow', 0))
    info['reds'].append(track_status_counts.get('Red', 0))
    info['sc'].append(track_status_counts.get('SCDeployed', 0))
    info['vsc'].append(track_status_counts.get('VSCDeployed', 0))