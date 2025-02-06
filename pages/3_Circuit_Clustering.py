import streamlit as st
import fastf1
import plotly.express as px
import pandas as pd

from src.dashboard.season import *

st.set_page_config(page_title="F1 Season report", page_icon="🏎️", layout="wide")
st.title("🏎️ Clustering de circuitos WEEEE")


with st.sidebar:
    season = st.selectbox("Select a season", list(range(2018, 2025)))

    if 'df' not in st.session_state:
        st.session_state.df = None

    if st.button("Load Data"):
        with st.spinner("Loading race data..."):

            st.session_state.df = pd.read_csv('data/output/featured_circuits_complete.csv', index_col=0)
            st.session_state.df_scaled = pd.read_csv('../data/preprocessed/circuits_scaled.csv', index_col=0)
            st.write('Data loaded')


if st.session_state.df is not None:

    st.dataframe(st.session_state.df.head())

    columns = st.multiselect('Hola', st.session_state.df_scaled.columns.to_list())
 
    df_kmeans = st.session_state.df_scaled[columns]