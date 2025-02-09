# Web application framework
# -----------------------------------------------------------------------
import streamlit as st

# FastF1 data handling
# -----------------------------------------------------------------------
import fastf1

# Data processing
# -----------------------------------------------------------------------
import pandas as pd

# Custom modules
# -----------------------------------------------------------------------
from src.dashboard.season import *


st.set_page_config(page_title="F1 Season report", page_icon="🏁", layout="wide")
st.title("🏎️ F1 Season Dashboard")

# Season selection
with st.sidebar:
    season = st.selectbox("Select a season", list(range(2018, 2025)))

    if 'df' not in st.session_state:
        st.session_state.clear()
        st.session_state.df = None

    if 'session' not in st.session_state:
        st.session_state.session = None

    if st.button("Load Data"):
        with st.spinner("Loading race data..."):

            try:
                st.session_state.df = pd.read_csv(f'data/seasons/{season}.csv', index_col=0)
                st.write('Data loaded')

            except:
                st.write("Getting results")
                st.session_state.df = get_results(season)
                st.write("Results ready")
                #df.to_csv(f'data/seasons/{season}.csv')

        st.session_state.session = fastf1.get_session(season, 1, 'R')
        st.session_state.session.load()
 

if st.session_state.df is not None:

    with st.sidebar:
        options = ["Drivers championship", "Constructors championship"]
        viz_type = st.selectbox("Select viz type", options)

    if viz_type == "Drivers championship":
        df_drivers = get_drivers_championship(st.session_state.df)
        fig = plot_drivers_championship(df_drivers, st.session_state.session, top=10)
        # fig = plot_standings_chart(df_drivers)
        st.plotly_chart(fig)

    elif viz_type == "Constructors championship":
        df_constructors = get_constructor_championship(st.session_state.df)
        fig = plot_constructors_championship(df_constructors, st.session_state.session, top=None)
        # fig = plot_standings_chart(df_constructors)
        st.plotly_chart(fig)