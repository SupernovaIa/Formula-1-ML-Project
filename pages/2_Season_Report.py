import streamlit as st
import fastf1
import plotly.express as px
import pandas as pd

from src.dashboard.season import *

st.set_page_config(page_title="F1 Season report", page_icon="🏎️", layout="wide")
st.title("🏎️ Visualizaciones de Carreras de F1 con FastF1 y Plotly")