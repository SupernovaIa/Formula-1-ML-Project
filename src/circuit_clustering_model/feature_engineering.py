import numpy as np

def generate_features(df):
    """
    Generates additional features for a given DataFrame by computing various proportions 
    and normalizing specific values related to track characteristics.

    Parameters
    -----------
    - df (pd.DataFrame): Input DataFrame containing track and performance data.

    Returns
    --------
    - (pd.DataFrame): DataFrame with new computed features and unnecessary columns removed.
    """
    df_featured = df.copy()

    # Avoid zero division (will not happen but just in case)
    df_featured['n_corners'] = df_featured['n_corners'].replace(0, np.nan)
    df_featured['distance'] = df_featured['distance'].replace(0, np.nan)

    # Add short and long gear number of corners proportion
    df_featured['short_gear_corners_prop'] = df_featured[['n_gear1_corners', 'n_gear2_corners', 'n_gear3_corners', 'n_gear4_corners']].sum(axis=1) / df_featured['n_corners']
    df_featured['long_gear_corners_prop'] = df_featured[['n_gear5_corners', 'n_gear6_corners', 'n_gear7_corners', 'n_gear8_corners']].sum(axis=1) / df_featured['n_corners']

    # Type of corners proportion
    df_featured[['slow_corners_prop', 'medium_corners_prop', 'fast_corners_prop']] = df_featured[['n_slow_corners', 'n_medium_corners', 'n_fast_corners']].div(df_featured['n_corners'], axis=0).fillna(0)

    # Other feartures proportion
    df_featured['straight_prop'] = df_featured['straight_length'] / df_featured['distance']
    df_featured['gear_changes_per_km'] = df_featured['gear_changes'] / df_featured['distance'] * 1000
    df_featured['n_corners_per_km'] = df_featured['n_corners'] / df_featured['distance'] * 1000

    # Fill NaN (just in case)
    df_featured.fillna(0, inplace=True)

    # Drop not-needed original columns
    cols_to_drop = [
        'n_gear1_corners', 'n_gear2_corners', 'n_gear3_corners', 'n_gear4_corners',
        'n_gear5_corners', 'n_gear6_corners', 'n_gear7_corners', 'n_gear8_corners',
        'gear_changes', 'straight_length', 'n_corners',
        'n_slow_corners', 'n_medium_corners', 'n_fast_corners',
        'laptime'
    ]

    df_featured.drop(columns=[col for col in cols_to_drop if col in df_featured.columns], inplace=True)

    return df_featured