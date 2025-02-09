# Notebook structure

- **A1-EDA-Clustering.ipynb**: Data exploration for circuit clustering, including visualization, pattern analysis, and feature engineering.
- **A2-Model-Clustering.ipynb**: Implementation and evaluation of clustering models for circuit segmentation.

- **B1-EDA-Classification.ipynb**: Data extraction and exploration for classification modeling, including feature selection and feature engineering.

- **B2-Model-Classification.ipynb**: Development and evaluation of classification models for race outcome predictions.

- **D1-Dashboard_Season.ipynb**: Loads FastF1 session data to analyze full F1 seasons. Results and conclusions must be analyzed for each season individually. This serves as a preview of the dashboard visualizations.

- **D2-Dashboard_Race.ipynb**: Loads FastF1 session data to analyze individual F1 races. Results and conclusions must be analyzed for each race separately. This serves as a preview of the dashboard visualizations.

# A1 - EDA Clustering

### Seasons considered
- The analysis covers races from **2018 to 2024**, excluding sprint sessions.
- Only circuits that appeared at least once in this period are included, resulting in **31 circuits initially**, later reduced to **30** due to data limitations.
- The timeframe was chosen because telemetry data has been available since 2018, ensuring a representative sample of modern F1 circuits.

### Source of information
- The **pole position lap** from qualifying was selected for analysis since it occurs under **optimal conditions** (new tires, low fuel, and maximum driver effort), making it more reliable than race laps.
- In wet qualifying sessions, an alternative lap from the most recent dry session was used. This adjustment affected **Spa and Interlagos**.
- The **Mugello circuit** was excluded due to missing corner data, as it hosted a Grand Prix only in **2020** as an exceptional case.

## Circuit clustering features

To cluster circuits effectively, key telemetry-derived characteristics were selected, ensuring that they provide a meaningful distinction between different track layouts:

- **Basic metrics**:
  - `laptime`: Total lap time in seconds.
  - `max_speed`: Maximum speed recorded in the speed trap (km/h).
  - `distance`: Total track length (meters).
  - `n_corners`: Total number of corners.
  - `gear_changes`: Total number of gear shifts during the lap.

- **Speed and handling**:
  - `avg_corner_speed`: Average speed through corners (km/h).
  - `avg_speed`: Average lap speed (km/h).
  - `throttle_perc`: Percentage of time spent applying throttle.
  - `brake_perc`: Percentage of time spent braking.

- **Track layout details**:
  - `straight_length`: Total length of straights exceeding 500 meters.
  - `n_slow_corners`: Corners with speeds below 120 km/h.
  - `n_medium_corners`: Corners with speeds between 120 km/h and 240 km/h.
  - `n_fast_corners`: Corners taken at 240 km/h or higher.
  - `n_gear{i}_corners`: Number of corners taken in each gear (1st to 8th).

> **Note:** The `compound` column (tire type) was included only to filter out wet sessions and was not used in clustering.

## Statistical summary and feature engineering

- Lap times ranged from **64.31s to 106.17s**, with an average of **82.86s**, showing the variety in circuit lengths and complexity.
- The **maximum speed** had a high variance, averaging **319.83 km/h**, ranging from **280 km/h to 347 km/h**.
- The **number of corners** varied significantly, from **10 to 27**, with an average of **16.5**.
- Correlation analysis revealed strong dependencies between variables like `throttle_perc`, `brake_perc`, and `avg_speed`, leading to the removal of redundant columns.

### Feature transformation

To ensure fair clustering, raw extensive metrics were transformed into intensive variables:
- **Normalized Corner Distribution:**
  - `short_gear_corners_prop`: Proportion of corners taken in **low gears** (1-4).
  - `long_gear_corners_prop`: Proportion of corners taken in **high gears** (5-8).
  - `slow_corners_prop`, `medium_corners_prop`, `fast_corners_prop`: Proportions of different cornering speeds.
- **Other Derived Features:**
  - `straight_prop`: Proportion of track occupied by straights.
  - `gear_changes_per_km`: Gear shifts per kilometer.
  - `n_corners_per_km`: Corners per kilometer.

These engineered features ensure that circuit characteristics are captured relative to their overall layout, making clustering results more meaningful.

# A2 - Clustering model

## Preprocessing
- The dataset with **feature engineering** was used for clustering to ensure meaningful variables.
- No categorical variables required encoding.
- A **MinMax scaler** was applied, as the dataset contained minimal outliers and no extreme values.

## Feature selection for clustering
- The selected variables for circuit segmentation were:
  - `avg_speed`: Capturing the overall speed profile of the circuit.
  - `straight_prop`: Measuring the proportion of straights relative to the total lap distance.
  - `slow_corners_prop`: Representing the percentage of slow corners.
- A **limited set of features** was used to avoid issues related to high dimensionality and ensure clear segmentation.
- Future iterations may involve testing additional features using **dimensionality reduction techniques** like `PCA` or feature importance analysis.

## K-Means clustering
- The **elbow method** initially suggested **3 clusters**, but due to the significant diversity among circuits, a more granular segmentation was necessary.
- After evaluating multiple cluster numbers and feature combinations, **7 clusters** were chosen as the optimal balance.
- Clustering quality was assessed using:
  - **Silhouette Score**: `0.3314`
  - **Davies-Bouldin Index**: `0.6889`

## Cluster visualization
- **Radar plots** were used to compare cluster profiles across multiple variables.
- **Scatter plots** were generated to visualize clustering patterns based on selected features.
- **Principal Component Analysis (PCA)** was performed to:
  - Reduce dimensionality while retaining key patterns.
  - Eliminate redundant information by removing correlated features.
  - Improve visualization and cluster interpretability.

## Alternative clustering methods
- **Agglomerative clustering** and **DBSCAN** were tested but did not yield clear or well-separated clusters.
- **K-Means** provided the best results in terms of balance between cohesion, separation, and interpretability.

# B1 - EDA Classification

## Dataset overview
- The dataset consists of **Formula 1 race results from the 2010 season to the present**.
- The starting point of 2010 was chosen due to significant **regulatory changes**, including:
  - Elimination of refueling, altering race strategy.
  - Introduction of a new scoring system emphasizing **consistency**.
  - Evolution of tire suppliers and car regulations.
- The dataset includes **305 races**, extracting results for each driver in every Grand Prix.
- The data has been retrieved using `FastF1`, which integrates multiple sources to ensure high accuracy.

## Key variables
The dataset includes crucial features to analyze race outcomes and driver performance:

- **`DriverId`**: Unique identifier for each driver.
- **`TeamId`**: Identifier of the team to which the driver belongs.
- **`Position`**: Final race position.
- **`GridPosition`**: Starting position on the grid.
- **`Time`**: Time difference with respect to the race winner.
- **`Status`**: Final status of the driver (finished, retired, etc.).
- **`Points`**: Points earned by the driver in the race.
- **`season`**: Season in which the Grand Prix took place.
- **`round`**: Round number within the season.
- **`circuitId`**: Unique identifier of the circuit where the race was held.

These features form the foundation for exploring patterns, evaluating performance, and predicting race outcomes.

## Data cleaning
- **Handling missing data:**
  - `GridPosition` missing values corresponded to drivers who did not qualify and were removed.
  - `Time` missing values were not relevant for classification and were excluded.
- **Standardizing team names:**
  - Several teams have changed names over time, requiring unification. The most recent names were retained for consistency.
  - Examples:
    - `force_india`, `racing_point`, `aston_martin` → `aston_martin`
    - `toro_rosso`, `alphatauri`, `rb` → `rb`
    - `renault`, `lotus_f1`, `alpine` → `alpine`

## Feature engineering
To improve model performance, additional derived variables were created:

- **`Winner`**: Binary variable indicating if the driver won the race (`Position = 1`).
- **`Podium`**: Indicates whether the driver finished in the **top 3**.
- **Performance-based features:**
  - **`MeanPreviousGrid`**: Average grid position over the last 3 races.
  - **`MeanPreviousPosition`**: Average finishing position over the last 3 races.
  - **`CurrentDriverPoints`**: Accumulated driver points before the race.
  - **`CurrentDriverWins`**: Number of race wins before the race.
  - **`CurrentDriverPodiums`**: Number of podium finishes before the race.
  - **`CurrentTeamPoints`**: Accumulated team points before the race.

### Feature selection
To remove redundant features, correlation analysis was conducted:
- **Highly correlated features (>0.9 correlation) were removed:**
  - `season`: It had no correlation with other variables.
  - `CurrentDriverPoints`, `CurrentTeamPoints`: These were highly correlated with podium finishes and race wins.
- **Kept `round` feature**: This is essential for contextualizing accumulated statistics within a season.

# B2 - Classification models

## Preprocessing
- **Feature Selection:** Removed `Position`, `Time`, `Status`, `Points`, and `Podium` to avoid data leakage.
- **Encoding:**
  - `DriverId` and `TeamId` used **target encoding** to emphasize their performance.
  - `circuitId` used **ordinal encoding** since circuits do not carry an inherent order beyond race performance impact.
- **Scaling:** Applied **MinMax scaling**, as the dataset had few outliers and no extreme values.

## Model selection
A **binary classification problem** was formulated to predict race winners, testing:
- **Logistic Regression**: A simple yet interpretable model.
- **XGBoost**: A powerful gradient boosting model.

## Model evaluation

### Logistic Regression
- **Accuracy:** 96%
- **AUC-ROC:** 0.95
- **Limitations:** Overfitting detected, with test precision dropping from 0.69 to 0.47 and recall from 0.46 to 0.37.
- **Feature Importance:** Starting grid position and recent performance history were the most relevant factors.

### XGBoost
- **Accuracy:** 97.4%
- **AUC-ROC:** 0.967 (better than Logistic Regression).
- **Strengths:** Higher precision (0.75 vs. 0.47) and a more balanced F1-score (0.593 vs. 0.414).
- **Weaknesses:** While false positives were reduced, recall remained a challenge (0.49 vs. 0.37 in Logistic Regression).
- **Feature Importance:** Previous finishing positions were more critical than current grid positions.

## Class balancing
- Formula 1 races have a **severe class imbalance**, with only one winner per 20 drivers (~5% of the dataset).
- Instead of upsampling/downsampling, a **filtering approach** was used:
  - Retained only competitive drivers with reasonable starting and finishing positions.
  - Ensured drivers with strong comebacks or poor performances after good grid positions were included.
- **Results:**
  - Logistic Regression benefited significantly from balancing.
  - XGBoost saw **no substantial improvement**, as tree-based models handle imbalances better internally.

## Final model selection
- **XGBoost with the full dataset** was chosen for the best performance.
- **Logistic Regression with a balanced dataset** was a strong alternative for computational efficiency.

# D1 & D2 - Dashboard

These notebooks serve as a foundation for conducting race and full-season analyses, enabling future studies and documentation.