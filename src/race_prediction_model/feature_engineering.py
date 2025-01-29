

def add_features_to_results(df, window=3):
    """
    Enhances a race results DataFrame by adding statistical features such as 
    previous performance metrics, cumulative points, and win/podium counts.

    Parameters
    -----------
    - df (pd.DataFrame): DataFrame containing race results with at least the 
    columns ['season', 'round', 'DriverId', 'TeamId', 'Position', 'GridPosition', 'Points'].
    - window (int, optional): The number of previous races to consider for rolling statistics. 
    Defaults to 3.

    Returns
    --------
    - (pd.DataFrame): The input DataFrame with additional feature columns:
    - 'Winner': 1 if the driver won the race, 0 otherwise.
    - 'Podium': 1 if the driver finished in the top 3, 0 otherwise.
    - 'MeanPreviousGrid': Mean of the driver's grid positions over the previous races.
    - 'MeanPreviousPosition': Mean of the driver's finishing positions over the previous races.
    - 'CurrentDriverPoints': Cumulative driver points before the race.
    - 'CurrentDriverWins': Cumulative race wins before the race.
    - 'CurrentDriverPodiums': Cumulative podium finishes before the race.
    - 'CurrentTeamPoints': Cumulative team points before the race.
    """

    # Sort the DataFrame by season and round (just in case it's not)
    df.sort_values(by=['season', 'round'], ascending=[True, True], inplace=True)

    # Add features
    df['Winner'] = df['Position'].apply(lambda x: int(x == 1))
    df['Podium'] = df['Position'].apply(lambda x: int(x in [1, 2, 3]))
    df['MeanPreviousGrid'] = df.groupby('DriverId')['GridPosition'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    df['MeanPreviousPosition'] = df.groupby('DriverId')['Position'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    df['CurrentDriverPoints'] = df.groupby(['season', 'DriverId'])['Points'].cumsum() - df['Points']
    df['CurrentDriverWins'] = df.groupby(['season', 'DriverId'])['Winner'].cumsum() - df['Winner']
    df['CurrentDriverPodiums'] = df.groupby(['season', 'DriverId'])['Podium'].cumsum() - df['Podium']
    df['CurrentTeamPoints'] = df.groupby(['season', 'TeamId'])['Points'].cumsum() - df.groupby(['season', 'TeamId', 'round'])['Points'].cumsum()