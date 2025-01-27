from scipy.interpolate import interp1d
import pandas as pd
import fastf1
from tqdm import tqdm



def get_nearest(curve_distance, car_data, column):
    """
    Interpolates the value of a specified column in a DataFrame based on a given distance.

    This function uses linear interpolation to estimate the value of a specified column 
    at a given distance (`curve_distance`) using the data provided in the `car_data` DataFrame. 
    It handles missing values and checks if the distance is within the range of available data.

    Parameters:
    -----------
    curve_distance (float): The distance at which the value needs to be interpolated.
    car_data (pandas.DataFrame): A DataFrame containing at least two columns: 'Distance' and the specified column for interpolation. Any rows with null values in these columns will be dropped before processing.
    column (str): The name of the column whose value is to be interpolated.

    Returns:
    --------
    -- (float): The interpolated value of the specified column at the given distance.
    """

    # Ensure there are no null values in the data
    car_data = car_data.dropna(subset=['Distance', column])

    # Validate that there is enough data
    if car_data.empty:
        raise ValueError(f"The car_data DataFrame does not contain valid data in 'Distance' or '{column}'.")

    # Check if the curve distance is out of range
    if curve_distance < car_data['Distance'].min() or curve_distance > car_data['Distance'].max():
        print(f"Warning: The distance {curve_distance} is out of range of the 'Distance' data.")

    # Create the interpolation function
    interp_func = interp1d(car_data['Distance'], car_data[column], bounds_error=False, fill_value="extrapolate")

    # Return the interpolated speed for the given distance
    return interp_func(curve_distance)


def categorize_speed(speed):
    if speed < 120:
        return "slow"
    elif 120 <= speed <= 240:
        return "medium"
    else:
        return "fast"
    

def get_qualy_lap(session):
    """
      Extracts pole position lap data from a FastF1 qualyfiyng session, including car performance and circuit corner information.

      Parameters
      -----------
      - session (fastf1.core.Session): The FastF1 session object containing lap and circuit data.

      Returns
      --------
      - (tuple): A tuple containing:
      - dc (dict): A dictionary with key metrics such as compound, lap time, max speed, average speeds, and more.
      - lap (fastf1.core.Lap): The fastest lap object from the session.
      - car_data (pandas.DataFrame): The car data with added distance information.
      - corners (pandas.DataFrame): DataFrame containing corner information with calculated speeds.
      """

    # Getting lap data
    lap = session.laps.pick_fastest()
    car_data = lap.get_car_data().add_distance()
    
    # Getting corners info
    circuit_info = session.get_circuit_info()
    corners = circuit_info.corners[['Number', 'Distance']].copy()

    # If car data is empty, raise an error
    if car_data.empty:
        raise ValueError("Car telemetry data is empty.")

    # If car data is empty, raise an error
    elif corners.empty:
        raise ValueError("Corner data is empty.")

    # Add speed and kind of corner
    corners['Speed'] = corners['Distance'].apply(lambda x: get_nearest(x, car_data, 'Speed'))
    corners['Category'] = corners['Speed'].apply(categorize_speed)
    corners['nGear'] = corners['Distance'].apply(lambda x: get_nearest(x, car_data, 'nGear'))

    # Count number of corners of every kind
    category_counts = corners["Category"].value_counts()

    # Add number of corners for each gear
    gear_counts = corners['nGear'].value_counts().to_dict()
    gear_counts_full = {f'n_gear{gear}_corners': gear_counts.get(gear, 0) for gear in range(1,9)}

    # Add straights info
    straights = corners['Distance'].diff()

    # Fix main straight length
    straights.iloc[0] = car_data['Distance'].max() - corners['Distance'].iloc[-1] + corners['Distance'].iloc[0]

    # Get total straight length
    st_threshold = 500 # 500 m for straight threshold
    straight_length = straights[straights > st_threshold].sum()

    # Build dictionary with relevant info
    dc = {
        'compound': lap.Compound,
        'laptime': lap.LapTime.total_seconds(),
        'max_speed': lap.SpeedST,
        'distance': car_data['Distance'].max(),
        'n_corners': corners['Number'].max(),
        'avg_corner_speed': corners['Speed'].mean(),
        'avg_speed': car_data['Speed'].mean(),
        'throttle_perc': car_data['Throttle'].mean(),
        'brake_perc': car_data['Brake'].mean() * 100,
        'straight_lenght': straight_length,
        'gear_changes': car_data['nGear'].diff().abs().sum(),
        'n_slow_corners': category_counts.get('slow', 0),
        'n_medium_corners': category_counts.get('medium', 0),
        'n_fast_corners': category_counts.get('fast', 0)
        }
    
    # Add gear counts
    dc.update(gear_counts_full)

    return dc, lap, car_data, corners


def get_circuit_info(season, rnd):
    """
    Retrieves circuit information for a specific race round in a given season using qualifying session data.

    Parameters
    -----------
    - season (int): The season (year) of the race.
    - rnd (int): The round of the race in the specified season.

    Returns
    --------
    - (tuple): A tuple containing:
    - dc (dict): A dictionary with key metrics such as compound, lap time, max speed, average speeds, and more.
    - lap (fastf1.core.Lap): The fastest lap object from the session.
    - car_data (pandas.DataFrame): The car data with added distance information.
    - corners (pandas.DataFrame): DataFrame containing corner information with calculated speeds.
    """

    # Load session
    session = fastf1.get_session(season, rnd, 'Q')
    session.load(telemetry=True, weather=False)

    # Return circuit info from the pole position lap
    dc, lap, car_data, corners = get_qualy_lap(session)
    return dc, lap, car_data, corners


def extract_races_and_results_dataframes(races):
    """
    Extract circuit information and create a DataFrame based on provided races data.

    Parameters
    -----------
    - races (pd.DataFrame): A DataFrame containing race details with required columns: 'season', 'round', and 'circuitId'.

    Returns
    --------
    - (pd.DataFrame): A DataFrame constructed from circuit information, indexed by circuit IDs.
    """

    # Verify if required columns exists
    required_columns = {'season', 'round', 'circuitId'}

    if not required_columns.issubset(races.columns):
        raise ValueError(f"The 'races' DataFrame must contain the columns {required_columns}.")

    # Select columns
    races = races.loc[:, ['season', 'round', 'circuitId']]

    # Dictionary to store circuit information
    circuits = {}

    # Iterate by races
    for season, rnd, circuit_id in tqdm(races.itertuples(index=False), total=len(races), desc="Processing circuits."):
        try:
            # Get circuit info
            circuits[circuit_id] = get_circuit_info(season, rnd)[0]
        except Exception as e:
            print(f"Error processing circuit {circuit_id} (season {season}, round {rnd}): {e}")

    # Create dataframe from circuits
    data = pd.DataFrame.from_dict(circuits, orient='index')

    return data