import pandas as pd
import plotly.express as px
from plotly.io import show
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np

from fastf1.ergast import Ergast
import fastf1.plotting
import fastf1


def get_results(season):
    """
    Fetches and processes race results for a given Formula 1 season.

    Parameters
    ----------
    - season (int): The season year for which to fetch race results.

    Returns
    -------
    - (pd.DataFrame): A DataFrame containing processed results for each race, with columns for driver and constructor details, race points, sprint points (if applicable), grid and position deltas, and race metadata.
    """

    ergast = Ergast()
    races = ergast.get_race_schedule(season)

    results = []

    # For each race in the season
    for rnd, race in races['raceName'].items():

        # Get results
        temp = ergast.get_race_results(season=season, round=rnd + 1).content[0]

        # If there is a sprint, get the results
        sprint = ergast.get_sprint_results(season=season, round=rnd + 1)

        if sprint.content and sprint.description['round'][0] == rnd + 1:

            # Merge sprint results
            temp = pd.merge(temp, sprint.content[0], on=['driverCode', 'constructorName'], how='left')

            # Add sprint points and race points to get the total
            temp['points'] = temp['points_x'] + temp['points_y']

            # Change names columns
            temp = temp.rename(columns={'grid_x': 'grid', 'position_x': 'position', 'grid_y': 'grid_sprint', 'position_y': 'position_sprint'})

            # Keep useful columns
            temp = temp[['constructorName', 'driverCode', 'points', 'grid', 'position', 'grid_sprint', 'position_sprint']]
            temp['delta_sprint'] = temp['grid_sprint'] - temp['position_sprint']

        # If there is not any sprint, we just set NaN 
        else:
            temp = temp[['constructorName', 'driverCode', 'points', 'grid', 'position']]
            temp['grid_sprint'] = np.nan
            temp['position_sprint'] = np.nan
            temp['delta_sprint'] = np.nan

        # Add round number and Grand Prix name
        temp['round'] = rnd + 1
        temp['race'] = race.removesuffix(' Grand Prix')
        temp['delta'] = temp['grid'] - temp['position']

        # Add to results
        results.append(temp)

    # Append all races into a single dataframe
    results = pd.concat(results)

    return results


def get_drivers_championship(df_results):
    """
    Generates a drivers' championship standings table based on race results.

    Parameters
    ----------
    - df_results (pd.DataFrame): DataFrame containing race results with columns `driverCode`, `round`, `points`, and `race`.

    Returns
    -------
    - (pd.DataFrame): A pivot table with drivers as rows, races as columns, and points as values, sorted by total points.
    """

    df = df_results.copy()

    # Get the results matrix by GP and driver
    df = df.pivot(index='driverCode', columns='round', values='points')

    # Rank the drivers by their total points
    df['total_points'] = df.sum(axis=1)
    df = df.sort_values(by='total_points', ascending=False)
    df.drop(columns='total_points', inplace=True)

    # Use race name, instead of round number, as column names
    df.columns = df_results['race'].drop_duplicates()

    return df


def get_constructor_championship(df_results):
    """
    Generates a constructors' championship standings table based on race results.

    Parameters
    ----------
    - df_results (pd.DataFrame): DataFrame containing race results with columns `round`, `race`, `constructorName`, and `points`.

    Returns
    -------
    - (pd.DataFrame): A pivot table with constructors as rows, races as columns, and points as values, sorted by total points.
    """

    df = df_results.copy()

    # We sum points for every constructor and GP
    df = df.groupby(['round', 'race', 'constructorName'])['points'].sum().reset_index()

    # Get the results matrix by GP and constructor
    df = df.pivot(index='constructorName', columns='round', values='points')

    # Rank the teams by their total points
    df['total_points'] = df.sum(axis=1)
    df = df.sort_values(by='total_points', ascending=False)
    df.drop(columns='total_points', inplace=True)

    # Use race name, instead of round number, as column names
    df.columns = df_results['race'].drop_duplicates()

    return df


def plot_standings_chart(df):
    """
    Generate and display a standings heatmap chart using the given DataFrame.

    Parameters
    ----------
    - df (pd.DataFrame): A pandas DataFrame where rows represent entities (e.g., drivers, teams) and columns represent races or events, with values indicating standings or points.

    Returns
    -------
    - None: Displays the generated heatmap chart.
    """

    tag = df.index.name

    fig = px.imshow(
        df, 
        text_auto=True, 
        aspect='auto', 
        color_continuous_scale='tempo',
        labels={'x': 'Race',
                'y': tag,
                'color': 'Points'}       # Hover texts
    )

    # Remove axis titles
    fig.update_xaxes(title_text='')
    fig.update_yaxes(title_text='')

    # Show all ticks, i.e. driver names
    fig.update_yaxes(tickmode='linear')

    # Show horizontal grid only
    fig.update_xaxes(showgrid=False, showline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', showline=False, tickson='boundaries')

    # Black background
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))   

    # Remove legend
    fig.update_layout(coloraxis_showscale=False)   

    # x-axis on top
    fig.update_layout(xaxis=dict(side='top'))

    show(fig)
