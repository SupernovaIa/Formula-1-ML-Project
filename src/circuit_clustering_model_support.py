from scipy.interpolate import interp1d


def get_nearest_speed(curve_distance, car_data):
    """
    Calculates the nearest interpolated speed at a given curve distance using car data.

    Parameters
    -----------
    - curve_distance (float): The distance of the curve for which the speed is to be calculated.
    - car_data (pandas.DataFrame): A DataFrame containing 'Distance' and 'Speed' columns, representing car telemetry data.

    Returns
    --------
    - (float): The interpolated speed at the specified curve distance.

    Raises
    --------
    - ValueError: If the car_data DataFrame does not contain valid data in 'Distance' or 'Speed'.
    """
    
    # Ensure there are no null values in the data
    car_data = car_data.dropna(subset=['Distance', 'Speed'])

    # Validate that there is enough data
    if car_data.empty:
        raise ValueError("The car_data DataFrame does not contain valid data in 'Distance' or 'Speed'.")

    # Check if the curve distance is out of range
    if curve_distance < car_data['Distance'].min() or curve_distance > car_data['Distance'].max():
        print(f"Warning: The distance {curve_distance} is out of range of the 'Distance' data.")

    # Create the interpolation function
    interp_func = interp1d(car_data['Distance'], car_data['Speed'], bounds_error=False, fill_value="extrapolate")

    # Return the interpolated speed for the given distance
    return interp_func(curve_distance)


def get_qualy_lap(session):
    """
      Extracts and analyzes qualifying lap data from a FastF1 session, including car performance and circuit corner information.

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

    # Aplicar la función al dataframe de curvas
    corners['Speed'] = corners['Distance'].apply(lambda x: get_nearest_speed(x, car_data))

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
        'gear_changes': car_data['nGear'].diff().abs().sum()
        }

    return dc, lap, car_data, corners