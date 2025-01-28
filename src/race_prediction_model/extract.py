import time
from ratelimit import limits, RateLimitException
import pandas as pd
from fastf1.ergast import Ergast
import os
import fastf1
from tqdm import tqdm


# Call limit: [CALLS] every [PERIOD] seconds
CALLS = 200
PERIOD = 3600

@limits(calls=CALLS, period=PERIOD)
def load_session(session, laps=True, telemetry=False, weather=False, messages=True):
    session.load(laps=laps, telemetry=telemetry, weather=weather, messages=messages)


def load_session_with_retry(session):
    while True:
        try:
            load_session(session)
            break

        except RateLimitException:
            print("Call limit reached. Waiting...")
            time.sleep(PERIOD)


def _get_weather_condition(compounds):
    """
    Determine the race condition based on the proportion of tire compounds.

    Parameters
    -----------
    - compounds (pd.Series): A pandas Series where keys represent tire types (e.g., 'WET', 'INTERMEDIATE', etc.), 
    and values represent the proportions of each tire type.

    Returns
    --------
    - (str): The classified race condition, which can be:
    - "Wet" if more than 75% of tires are wet or intermediate.
    - "Mixed" if the proportion of wet or intermediate tires is between 20% and 75%.
    - "Dry" if the proportion of wet or intermediate tires is 20% or less.
    """

    # Thresholds to determine race conditions
    wet_threshold = 0.75
    mixed_threshold = 0.20

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
    

def _get_race_extra_info(session, dc):
    """
    Extracts additional information from a FastF1 session object and updates the provided info dictionary with metrics such as weather conditions, yellow flags, red flags, and safety car deployments.

    Parameters
    -----------
    - session (fastf1.core.Session): The FastF1 session object containing lap and track status data. Must include a 'laps' DataFrame with a 'Compound' column and a 'track_status' DataFrame with a 'Message' column.
    - dc (dict): A dictionary where the keys 'weather', 'yellows', 'reds', 'sc', and 'vsc' must be present. Each key should map to a list that will be updated with the corresponding session data.

    Returns
    --------
    - None: Updates the `info` dictionary in place.
    """

    # Check keys
    if not all(key in dc for key in ['weather', 'yellows', 'reds', 'sc', 'vsc']):
        raise ValueError("The 'info' dictionary must contain the following keys: 'weather', 'yellows', 'reds', 'sc', 'vsc'")
    
    # Check columns exists
    if 'Compound' not in session.laps.columns or 'Message' not in session.track_status.columns:
        raise KeyError("The columns 'Compound' or 'Message' are not available in the session data.")
    
    # Get compounds and track status
    compounds = session.laps['Compound'].value_counts(normalize=True)
    track_status_counts = session.track_status['Message'].value_counts()

    # Get metrics
    dc['weather'].append(_get_weather_condition(compounds))
    dc['yellows'].append(track_status_counts.get('Yellow', 0))
    dc['reds'].append(track_status_counts.get('Red', 0))
    dc['sc'].append(track_status_counts.get('SCDeployed', 0))
    dc['vsc'].append(track_status_counts.get('VSCDeployed', 0))


def _get_race_results(session, margin=100):
    """
    Retrieves race results and processes the data to standardize timing and statuses.

    Parameters
    -----------
    - session (fastf1.core.Session): The FastF1 session object containing race data, including results.
    - margin (int, optional): The time margin (in seconds) added to non-finishers' times. Defaults to 100.

    Returns
    --------
    - (pd.DataFrame): A DataFrame containing processed race results with the following columns:
        - 'DriverId': Identifier for the driver.
        - 'TeamId': Identifier for the team.
        - 'Position': Final position of the driver.
        - 'GridPosition': Starting grid position of the driver.
        - 'Time': Race time in seconds (adjusted for winners and non-finishers).
        - 'Status': Status of the driver (e.g., 'Finished', 'Retired').
        - 'Points': Points scored by the driver.
    """

    # Get results dataframe
    results = session.results
    results = results.loc[:, ['DriverId', 'TeamId', 'Position', 'GridPosition', 'Time', 'Status', 'Points']]

    # Fix winner time to 0 and convert to seconds
    results.iloc[0, results.columns.get_loc('Time')] = pd.Timedelta(0)
    results['Time'] = results['Time'].dt.total_seconds()

    # Fix non-finishers time to avoid NaT
    max_finished_time = results.loc[results['Status'] == 'Finished', 'Time'].max()
    results.loc[results['Status'] != 'Finished', 'Time'] = max_finished_time + margin

    return results


def extract_races_dataframe(start, end=None, save=True):
    """
    Extracts race schedules for a range of seasons and optionally saves the data to a CSV file.

    Parameters
    -----------
    - start (int): The starting season (year) for which race schedules will be extracted.
    - end (int, optional): The ending season (year) for which race schedules will be extracted. Defaults to `start` if not provided.
    - save (bool, optional): Whether to save the resulting dataframe as a CSV file. Defaults to `True`.

    Returns
    --------
    - (pd.DataFrame): A dataframe containing race schedules, including columns for season, round, and circuitId.
    """

    # If end not provided, we only get one season
    if not end:
        end = start

    # Initialize an Ergast object
    ergast = Ergast()

    # Dataframe to store races
    races_final_df = pd.DataFrame()

    # Iterate by seasons
    for i in tqdm(range(start, end + 1), desc='Loading seasons'):

        # Get season schedule
        races = ergast.get_race_schedule(i)
        races = races.loc[:, ['season', 'round', 'circuitId']]

        # Concatenate with previous results
        races_final_df = pd.concat([races_final_df, races], ignore_index=True)

    # Saving dataframe
    if save:
        output_dir = os.path.join('..', 'data', 'output') if 'notebook' in os.getcwd() else os.path.join('data', 'output')
        os.makedirs(output_dir, exist_ok=True)

        path_races = os.path.join(output_dir, 'races.csv')

        print(f"Saving races in {path_races}")
        races_final_df.to_csv(path_races, index=False)

    return races_final_df


def extract_results_dataframe(races_df, save=True):
    """
    Extracts race results from a dataframe of races and saves the results if specified.

    This function iterates through a dataframe of races, loads race session data for each season and round, retrieves race results, and compiles them into a final dataframe. Optionally, the results can be saved as a CSV file.

    Parameters
    -----------
    - races_df (pd.DataFrame): A dataframe containing race information, including season, round, and circuit ID.
    - save (bool): A flag indicating whether to save the final results dataframe as a CSV file. Default is True.

    Returns
    --------
    - (pd.DataFrame): A dataframe containing the combined race results for all processed races.
    - (dict): A dictionary of loaded race sessions, keyed by (season, round).
    """

    # Dataframe to store results
    results_final_df = pd.DataFrame()

    # Dictionary that stores sessions
    sessions = {}

    # Iterate by round within a season
    for season, rnd, circuit_id in tqdm(races_df.itertuples(index=False), desc='Processing results.'):

        try:
            # Load session
            print(f"Loading {season} season. Round: {rnd}...")
            session = fastf1.get_session(season, rnd, 'R')
            load_session_with_retry(session)

            # Save session in sessions dictionary
            sessions[(season, rnd)] = session

            # Get results dataframe
            results = _get_race_results(session)
            results['season'] = season
            results['round'] = rnd
            results['circuitId'] = circuit_id

            # Concatenate with previous results
            results_final_df = pd.concat([results_final_df, results])

        except Exception as e:
            print(f"Failed to load season {season}, round {rnd}: {e}")
        
    # Saving dataframes
    if save:
        output_dir = os.path.join('..', 'data', 'output') if 'notebook' in os.getcwd() else os.path.join('data', 'output')
        os.makedirs(output_dir, exist_ok=True)

        path_results = os.path.join(output_dir, 'results.csv')

        print(f"Saving results in {path_results}")
        results_final_df.to_csv(path_results, index=False)

    return results_final_df, sessions