import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

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


def plot_drivers_pace(session, kind='driver', threshold=None, box=False):
    """
    Plots the pace of drivers in a racing session, visualizing lap time distributions
    for the top 10 drivers (point finishers). The visualization can be customized
    to group data by driver or tire compound, with an optional box plot overlay.

    Parameters
    -----------
    - session (fastf1.core.Session): The session object containing data for a specific race session.
    - kind (str, optional): Determines the grouping of the visualization. Acceptable values are:
        - 'driver': Groups by driver (default).
        - 'compound': Groups by tire compound.
    - threshold (float, optional): A lap time threshold in seconds. Laps slower than this
        threshold are excluded from the plot. Default is None (no threshold applied).
    - box (bool, optional): Whether to include a box plot overlay on the violin plot. 
        Default is False (no box plot overlay).

    Returns
    --------
    - (None): Displays a Plotly violin plot of lap time distributions.
    """

    # Retrieve the top 10 drivers from the session (point finishers)
    point_finishers = session.drivers[:10]
    
    # Filter fast laps of the top 10 drivers, applying an optional lap time threshold
    driver_laps = session.laps.pick_drivers(point_finishers).pick_quicklaps(threshold=threshold).reset_index()

    # Get the finishing order using driver abbreviations
    finishing_order = [session.get_driver(i)["Abbreviation"] for i in point_finishers]

    # Convert lap times to seconds for better visualization
    driver_laps['LapTime(s)'] = driver_laps['LapTime'].dt.total_seconds()

    # Determine coloring and title based on the `kind` argument
    if kind.lower() == 'driver':
        color = "Driver"
        color_map = fastf1.plotting.get_driver_color_mapping(session=session)
        title = f"{session.event.year} {session.event.EventName} Lap Time Distributions by driver"

    elif kind.lower() == 'compound':
        color = 'Compound'
        color_map = fastf1.plotting.get_compound_mapping(session)
        title = f"{session.event.year} {session.event.EventName} Lap Time Distributions by driver and compound"

    # Create a violin plot
    fig = px.violin(
        driver_laps,
        x="Driver",
        y="LapTime(s)",
        color=color,
        box=box,
        points='all',
        hover_data=driver_laps,
        category_orders={"Driver": finishing_order},
        color_discrete_map=color_map
    )

    # Customize the layout of the plot
    fig.update_layout(
        title=title,
        xaxis_title="Driver",
        yaxis_title="Lap Time (s)",
        showlegend=True,
        template='plotly_dark'
    )

    fig.show()


def plot_teams_pace(session, kind='team', threshold=None, box=False):
    """
    Plots lap time distributions for teams or compounds in a racing session.

    This function creates a violin plot of lap time distributions for a racing session. 
    The data can be grouped by teams or tire compounds, with options to include box plots 
    and filter laps based on a lap time threshold.

    Parameters
    -----------
    - session (Session): The session object containing lap data and event details.
    - kind (str, optional): The type of grouping for the plot. Can be 'team' to group by teams 
    or 'compound' to group by tire compounds. Defaults to 'team'.
    - threshold (float, optional): A lap time threshold in seconds to filter out slower laps. 
    If None, no threshold is applied. Defaults to None.
    - box (bool, optional): If True, includes a box plot within the violin plot. Defaults to False.

    Returns
    --------
    - None: The function displays the plot directly.
    """

    # Filter fast laps, applying an optional lap time threshold
    team_laps = session.laps.pick_quicklaps(threshold=threshold).reset_index()

    # Convert lap times to seconds for better visualization
    team_laps['LapTime(s)'] = team_laps['LapTime'].dt.total_seconds()

    # Order the team from the fastest (lowest median lap time) to slower
    team_order = (
        team_laps[["Team", "LapTime(s)"]]
        .groupby("Team")
        .median()["LapTime(s)"]
        .sort_values()
        .index
    )

    # Make a color palette associating team names to hex codes
    team_palette = {team: fastf1.plotting.get_team_color(team, session=session)
                    for team in team_order}

    if kind.lower() == 'team':
        color = "Team"
        color_map = team_palette
        title = f"{session.event.year} {session.event.EventName} Lap Time Distributions by team"

    elif kind.lower() == 'compound':
        color = 'Compound'
        color_map = fastf1.plotting.get_compound_mapping(session)
        title = f"{session.event.year} {session.event.EventName} Lap Time Distributions by team and compound"

    # Create a violin plot
    fig = px.violin(
        team_laps,
        x="Team",
        y="LapTime(s)",
        color=color,
        box=box,
        points='all',
        hover_data=team_laps,
        category_orders={"Team": team_order},
        color_discrete_map=color_map
    )

    # Customize the layout of the plot
    fig.update_layout(
        title=title,
        xaxis_title="Constructor",
        yaxis_title="Lap Time(s)",
        showlegend=True,
        template='plotly_dark'
    )

    fig.show()


def plot_tyre_strat(session):
    """
    Plots the tyre strategies for a given race session.

    This function visualizes the stint lengths and compound choices for each driver in a race session,
    using a horizontal bar chart. Each bar represents a stint, color-coded by the compound used.

    Parameters
    -----------
    - session (fastf1.core.Session): The race session object containing event and lap data.

    Returns
    --------
    - None: Displays an interactive plotly visualization of the tyre strategies.
    """

    title = f"{session.event.year} {session.event.EventName} Strategies"
    shown_compounds = set()

    # Get drivers
    drivers = session.drivers
    drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]

    # Get stints
    laps = session.laps

    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()

    stints = stints.rename(columns={"LapNumber": "StintLength"})

    # Create plot
    fig = go.Figure()

    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver]
        previous_stint_end = 0

        for _, row in driver_stints.iterrows():

            compound_color = fastf1.plotting.get_compound_color(row["Compound"], session=session)

            # Show compound on legend only once
            if row["Compound"] not in shown_compounds:
                show_legend = True
                # Flag compound as already shown
                shown_compounds.add(row["Compound"])  

            else:
                show_legend = False

            fig.add_trace(
                go.Bar(
                    y=[driver],
                    x=[row["StintLength"]],
                    base=previous_stint_end,
                    orientation='h',
                    marker=dict(color=compound_color, line=dict(color='black', width=1)),
                    name=row["Compound"],
                    legendgroup=row["Compound"],
                    showlegend=show_legend,
                    width=0.8
                )
            )
            
            previous_stint_end += row["StintLength"]

    # Customize plot layout
    fig.update_layout(
        title=title,
        xaxis=dict(title="Lap Number"),
        yaxis=dict(title="", autorange='reversed'),
        plot_bgcolor='black',
        template='plotly_dark',
        margin=dict(t=40, b=40, l=40, r=40),
        bargap=1,
    )

    # Hide grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.show()


def plot_telemetry(session, mode='Speed', drivers=[]):
    """
    Plots telemetry data for a given race session.

    This function visualizes telemetry data (Speed, Throttle, or RPM) along the track's distance 
    for the fastest lap(s) of specified drivers. Vertical dashed lines and labels mark the corners 
    on the circuit.

    Parameters
    -----------
    - session (fastf1.core.Session): The race session object containing event, lap, and circuit data.
    - mode (str, optional): The telemetry data to plot. Must be one of ['Speed', 'Throttle', 'RPM']. 
    Defaults to 'Speed'.
    - drivers (list of str, optional): List of driver abbreviations to include in the plot. 
    If empty, only the session's fastest lap is plotted. Defaults to [].

    Returns
    --------
    - None: Displays an interactive plotly visualization of the telemetry data.
    """

    valid_modes = ['Speed', 'Throttle', 'RPM']

    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")


    circuit_info = session.get_circuit_info()
    title = f"{session.event.year} {session.event.EventName} Lap comparison"

    # Create plot
    fig = go.Figure()

    # Add a trace for every driver
    for driver in drivers:
        fastest_lap = session.laps.pick_drivers([driver]).pick_fastest()
        car_data = fastest_lap.get_car_data().add_distance()
        team_color = fastf1.plotting.get_team_color(fastest_lap['Team'], session=session)

        # Add main trace (speed, throttle or RPM vs distance)
        fig.add_trace(go.Scatter(
            x=car_data['Distance'],
            y=car_data[mode],
            mode='lines',
            line=dict(color=team_color),
            name=fastest_lap['Driver']
        ))

    # If no drivers, pick fastest
    if drivers == []:
        title = f"{session.event.year} {session.event.EventName} Pole Lap"
        fastest_lap = session.laps.pick_fastest()
        car_data = fastest_lap.get_car_data().add_distance()
        team_color = fastf1.plotting.get_team_color(fastest_lap['Team'], session=session)

        # Add main trace (speed, throttle or RPM vs distance)
        fig.add_trace(go.Scatter(
            x=car_data['Distance'],
            y=car_data[mode],
            mode='lines',
            line=dict(color=team_color),
            name=fastest_lap['Driver']
        ))

    # Vertical lines in corners
    v_min = car_data[mode].min()
    v_max = car_data[mode].max()

    for _, corner in circuit_info.corners.iterrows():
        # Add vertical line
        fig.add_trace(go.Scatter(
            x=[corner['Distance'], corner['Distance']],
            y=[v_min - 20, v_max + 20],
            mode='lines',
            line=dict(dash='dot', color='grey'),
            showlegend=False
        ))
        # Number of corner
        txt = f"{corner['Number']}{corner['Letter']}"
        fig.add_trace(go.Scatter(
            x=[corner['Distance']],
            y=[v_min - 30],
            mode='text',
            text=[txt],
            textposition='top center',
            showlegend=False
        ))

    # Customize plot layout
    fig.update_layout(
        title=title,
        xaxis_title='Distance (m)',
        yaxis_title=f'{mode} {"(km/h)" if mode == "Speed" else ""}',
        yaxis=dict(range=[v_min - 40, v_max + 20]),
        plot_bgcolor='black',
        template='plotly_dark',
        showlegend=True
    )

    # Hide grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.show()