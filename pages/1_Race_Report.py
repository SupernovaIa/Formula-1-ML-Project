import streamlit as st
import fastf1
import plotly.express as px
import pandas as pd

from src.dashboard.race import *

st.set_page_config(page_title="F1 Race report", page_icon="🏎️", layout="wide")
st.title("🏎️ Visualizaciones de Carreras de F1 con FastF1 y Plotly")

# Load available data
df = pd.read_csv('data/output/races.csv')

# Season and round selection
with st.sidebar:
    year = st.selectbox("Select a season", list(range(2018, 2025)))

    # Set a maximum
    max_ = df[df['season'] == year]['round'].max()

    round_number = st.number_input(f"Round number", min_value=1, max_value=max_, step=1)

    # Display circuit id
    circuit_id = df[(df['season'] == year) & (df['round'] == round_number)]['circuitId'].values[0]
    st.write("You selected:", circuit_id)

    # Session selection (either race or qualy)
    session = st.selectbox("Select a session", ['Qualifying', 'Race'])

    if 'session' not in st.session_state:
        st.session_state.session = None

    if st.button("Load Data"):
        with st.spinner("Loading race data..."):
            st.session_state.session = fastf1.get_session(year, round_number, session)
            st.session_state.session.load()
       

if st.session_state.session is not None:

    if session == 'Qualifying':

        with st.sidebar:
            options = ["Qualy results", "Telemetry", "Lap comparison"]
            viz_type = st.selectbox("Selecciona el tipo de visualización", options)

        if viz_type == "Qualy results":

            df = get_qualy_results(st.session_state.session)
            st.dataframe(df)

            fig = plot_fastest_laps(st.session_state.session)
            st.plotly_chart(fig)

        elif viz_type == "Telemetry":
            st.plotly_chart(draw_track(st.session_state.session))
            fig = plot_telemetry(st.session_state.session)
            st.plotly_chart(fig)

        elif viz_type == "Lap comparison":
            res = st.session_state.session.results
            mode = st.selectbox("Select visualization type", ['Speed', 'RPM', 'Throttle'])
            st.plotly_chart(plot_telemetry(st.session_state.session, mode, drivers=res['Abbreviation'].to_list()))

        

    elif session == 'Race':

        with st.sidebar:
            options = ["Results", "Position changes", "Driver Pace", "Pace", "Tyre strategies"]
            viz_type = st.selectbox("Selecciona el tipo de visualización", options)

        if viz_type == "Results":

            df = get_race_results(st.session_state.session).reset_index(drop=True)
            st.dataframe(df)

            fig = plot_results(st.session_state.session)
            st.plotly_chart(fig)

        elif viz_type == "Position changes":
            fig = plot_position_changes(st.session_state.session)
            st.plotly_chart(fig)

        elif viz_type == "Driver Pace":
            res = st.session_state.session.results
            driver_name = st.selectbox("Select a driver", res['FullName'].to_list())
            driver_abb = st.session_state.session.results.loc[st.session_state.session.results['FullName'] == driver_name, 'Abbreviation'].values[0]
            threshold = st.slider("Select threshold:", 100, 200, 107, 1) / 100
            fig = plot_driver_pace(st.session_state.session, driver_abb, threshold)
            st.plotly_chart(fig)

        elif viz_type == "Pace":
            kind = st.selectbox("Selecciona el tipo de visualización:", ["driver", "compound"])
            threshold = st.slider("Select threshold:", 100, 200, 107, 1) / 100
            box = st.checkbox("Mostrar boxplot", value=False)
            fig = plot_drivers_pace(st.session_state.session, kind, threshold, box)
            st.plotly_chart(fig)

        elif viz_type == "Tyre strategies":
        
            fig = plot_tyre_strat(st.session_state.session)
            st.plotly_chart(fig)