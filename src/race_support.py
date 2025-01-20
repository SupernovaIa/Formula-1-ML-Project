import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import fastf1.plotting

# Load FastF1's dark color scheme
# fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False, color_scheme='fastf1')

import plotly.graph_objects as go
import fastf1.plotting


def plot_position_changes(session):

    """
    Plots position changes of drivers during a racing session using Plotly.

    Parameters
    -----------
    - session (fastf1.core.Session): The session object containing lap and driver data.

    Returns
    --------
    - None: This function does not return a value; it displays a Plotly figure showing position changes.
    """

    # Get event information
    title = session.event.EventName
    year = session.event.EventDate.year
    rnd = session.event.RoundNumber

    # Linestyle mapping
    linestyle_map = {
        'solid': 'solid',
        'dashed': 'dash',
        'dotted': 'dot',
        'dashdot': 'dashdot',
        'longdash': 'longdashdot'
    }

    fig = go.Figure()

    for drv in session.drivers:
        drv_laps = session.laps.pick_drivers([drv])

        try:
            abb = drv_laps['Driver'].iloc[0]
            style = fastf1.plotting.get_driver_style(identifier=abb,
                                                     style=['color', 'linestyle'],
                                                     session=session)

            # Convert linestyle to Pltly format
            plotly_dash_style = linestyle_map.get(style['linestyle'], 'solid')

            fig.add_trace(go.Scatter(
                x=drv_laps['LapNumber'],
                y=drv_laps['Position'],
                mode='lines',
                name=abb,
                line=dict(color=style['color'], dash=plotly_dash_style)
            ))

        except IndexError:
            print('Driver not found. Probably DNS')

    # Dark theme and color configuration
    fig.update_layout(
        title=f'Position changes | Round {rnd} {title} {year}',
        xaxis_title='Lap',
        yaxis_title='Position',
        yaxis=dict(
            autorange='reversed',
            tickvals=[1, 5, 10, 15, 20],
            gridcolor='rgba(255, 255, 255, 0.1)',
        ),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        legend=dict(
            x=1.02, y=1, borderwidth=1,
            bgcolor='rgba(0, 0, 0, 0.5)',
            font=dict(color='white')
        ),
        margin=dict(t=50, r=150, b=50, l=50),
        width=1200,
        height=800,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    fig.show()


def get_race_results(session):
    """
    Extracts and formats race results from a given session.

    Parameters
    -----------
    - session (fastf1.core.Session): The session object containing race results data.

    Returns
    --------
    - (pandas.DataFrame): A DataFrame containing formatted race results with selected columns and calculated delta positions.
    """
    
    df = session.results.copy()

    # Get delta position
    df['DeltaPosition'] = df['GridPosition'] - df['Position']

    # Select relevant columns
    df = df.loc[:, ['ClassifiedPosition', 
                          'FullName', 
                          'DriverNumber',
                          'CountryCode', 
                          'TeamName', 
                          'Position', 
                          'GridPosition',
                          'DeltaPosition',
                          'Time', 
                          'Status',
                          'Points']]
    
    # Time formatting
    df['Time'] = df.loc[:, 'Time'].astype(str).str.replace('0 days ', '').str.replace(r'(00:)+', '+', regex=True).str.removesuffix('000')

    return df


def get_qualy_results(session):
    """
    Extracts and formats qualifying session results from a given session.

    Parameters
    -----------
    - session (fastf1.core.Session): The session object containing qualifying results data.

    Returns
    --------
    - (pandas.DataFrame): A DataFrame containing formatted qualifying results with selected columns and formatted Q1, Q2, and Q3 times.
    """

    df = session.results.copy()

    # Select relevant columns
    df = df.loc[:, ['Position', 'FullName', 'DriverNumber', 'CountryCode', 'TeamName', 'Q1', 'Q2', 'Q3']]

    # Time formatting
    format_time = lambda col: col.astype(str).str.replace('0 days ', '').str.replace(r'(00:)+', '', regex=True).str.removesuffix('000')
    df[['Q1', 'Q2', 'Q3']] = df.loc[:, ['Q1', 'Q2', 'Q3']].apply(format_time)

    return df