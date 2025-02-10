import streamlit as st
import joblib
import pandas as pd

# Title and description
st.title("🏠 Race winner prediction using ML 🔮")
st.write("Use this app to predict future 🚀")

# Load saved encoder & scaler
encoder = joblib.load("model/encoder.pkl")
scaler = joblib.load("model/scaler.pkl")

# Load model
model = joblib.load("model/best_model.pkl")


# Load available data
df = pd.read_csv('data/output/races.csv')
df_results = df = pd.read_csv('data/output/results.csv')

# Season and round selection
with st.sidebar:
    year = st.selectbox("Select a season", list(range(2018, 2025)))

    # Set a maximum
    max_ = df[df['season'] == year]['round'].max()

    round_number = st.number_input(f"Round number", min_value=1, max_value=max_, step=1)

    # Display circuit id
    circuit_id = df[(df['season'] == year) & (df['round'] == round_number)]['circuitId'].values[0]
    st.write("You selected:", circuit_id)

# Mask
mask = (df_results['season'] == year) & (df_results['round'] == round_number)

# Category options
drivers = df_results[mask]['DriverId'].unique().tolist()
teams = df_results[mask]['TeamId'].unique().tolist()

# Forms
st.header("🔧 Features")
col1, col2, col3, col4 = st.columns(4)
col5, col6 = st.columns(2)

with col1:
    DriverId = st.selectbox("Driver", drivers)

with col2:
    TeamId = st.selectbox("Team", teams)

with col3:
    MeanPreviousGrid = st.number_input("Previous grid position", min_value=1, max_value=20, value=5, step=1, help="Mean qualifying results in the previous 3 races.")

with col4:
    MeanPreviousPosition = st.number_input("Previous position", min_value=1, max_value=20, value=5, step=1, help="Mean race results in the previous 3 races.")

with col5:
    GridPosition = st.slider("Grid position", min_value=1, max_value=20, value=5, step=1, help="Select the starting position.")

with col6:
    if round_number == 1:
        CurrentDriverWins, CurrentDriverPodiums = 0, 0
        st.warning("Current driver wins and podiums are set to 0 in round 1.")

    else:
        CurrentDriverWins = st.slider("Current wins", min_value=0, max_value=round_number-1, value=0, step=1)

        if CurrentDriverWins == round_number-1:
            CurrentDriverPodiums = CurrentDriverWins
            st.warning(f"Current driver podiums is set to {str(CurrentDriverPodiums)}.")

        else:
            CurrentDriverPodiums = st.slider("Current podiums", min_value=CurrentDriverWins, max_value=round_number-1, value=CurrentDriverWins, step=1)


new_data = pd.DataFrame({
    'DriverId': [DriverId],
    'TeamId': [TeamId],
    'GridPosition': [GridPosition],
    'round': [round_number],
    'MeanPreviousGrid': [MeanPreviousGrid],
    'MeanPreviousPosition': [MeanPreviousPosition],
    'CurrentDriverWins': [CurrentDriverWins],
    'CurrentDriverPodiums': [CurrentDriverPodiums],
    'circuitId': [circuit_id]
})

df_encoded = encoder.transform(new_data)
df_scaled = scaler.transform(df_encoded)

df_scaled = pd.DataFrame(df_scaled, columns=df_encoded.columns, index=df_encoded.index)

# Make predictions
predictions = model.predict(df_scaled)
prob = model.predict_proba(df_scaled)

# Get the probability of winning (class 1)
win_probability = prob[:, 1]  # Assuming the second column corresponds to winning

# Display results
if predictions[0] == 1:
    st.success(f"Expected victory with a probability of {win_probability[0]:.2%}!")
else:
    st.error(f"The probability of winning is only {win_probability[0]:.2%}")