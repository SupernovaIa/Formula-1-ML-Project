


def add_features_to_results(results):
    """
    Add derived features to the race results DataFrame, including cumulative points, wins, and podiums for drivers and teams.

    Parameters
    -----------
    - results (pd.DataFrame): A DataFrame containing race results with columns 'season', 'round', 'DriverId', 'TeamId', 'Points', and 'Position'.

    Returns
    --------
    - None: The function modifies the input DataFrame in place by adding new feature columns:
    - 'DriverPointsCumulative': Cumulative points for each driver by season.
    - 'TeamPointsCumulative': Cumulative points for each team by season.
    - 'Winner': Indicates if the driver won the race (1 for win, 0 otherwise).
    - 'Podium': Indicates if the driver finished in the top 3 (1 for podium, 0 otherwise).
    - 'WinsCumulative': Cumulative race wins for each driver by season.
    - 'PodiumsCumulative': Cumulative podium finishes for each driver by season.
    """

    # Sort the DataFrame by season and round
    results.sort_values(by=['season', 'round'], ascending=[True, True], inplace=True)

    # Calculate cumulative points for each driver by season
    results['DriverPointsCumulative'] = results.groupby(['season', 'DriverId'])['Points'].cumsum()

    # Calculate cumulative points for each team by season
    results['TeamPointsCumulative'] = results.groupby(['season', 'TeamId'])['Points'].cumsum()

    # Determine if the driver won the race (Position 1)
    results['Winner'] = results['Position'].apply(lambda x: int(x == 1))

    # Determine if the driver finished on the podium (Position 1, 2, or 3)
    results['Podium'] = results['Position'].apply(lambda x: int(x in [1, 2, 3]))

    # Calculate cumulative wins for each driver by season
    results['WinsCumulative'] = results.groupby(['season', 'DriverId'])['Winner'].cumsum()

    # Calculate cumulative podiums for each driver by season
    results['PodiumsCumulative'] = results.groupby(['season', 'DriverId'])['Podium'].cumsum()
