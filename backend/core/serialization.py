import json

import pandas as pd
from plotly.graph_objects import Figure


def fig_to_json(fig: Figure):
    """Plotly figures carry numpy/pandas types FastAPI's default encoder
    can't handle, so we round-trip through Plotly's own JSON encoder."""
    return json.loads(fig.to_json())


def df_to_records(df: pd.DataFrame):
    return json.loads(df.to_json(orient="records"))
