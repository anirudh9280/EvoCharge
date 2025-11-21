# EV Charging Sessions Dataset

## Overview

This dataset contains 3,500 electric vehicle charging session records with detailed information about user behavior, charging patterns, and energy consumption.

## Dataset Source

- **Source**: Kaggle - Electric Vehicle Charging Sessions Dataset
- **Author**: zyan1999
- **File**: `ev_charging_sessions.csv`
- **Records**: 3,500 charging sessions
- **Columns**: 10 features

## Dataset Schema

| Column       | Type     | Description                                                              |
| ------------ | -------- | ------------------------------------------------------------------------ |
| session_id   | string   | Unique identifier for each charging session (e.g., CS0001)               |
| user_id      | string   | Unique identifier for the user (e.g., U339)                              |
| vehicle_id   | string   | Unique identifier for the vehicle (e.g., V347)                           |
| station_id   | string   | Unique identifier for the charging station (e.g., S091)                  |
| start_time   | datetime | Timestamp when charging session began                                    |
| end_time     | datetime | Timestamp when charging session ended                                    |
| duration_min | integer  | Duration of charging session in minutes (30-120 min)                     |
| energy_kWh   | float    | Total energy consumed during session in kilowatt-hours (7.89-113.36 kWh) |
| session_day  | string   | Day type: Weekday or Weekend                                             |
| session_type | string   | Type of charging session: Regular, Occasional, or Emergency              |

## Purpose and Usage

### Primary Use: Machine Learning Model Training

This dataset serves as the foundation for building a predictive machine learning model that estimates energy consumption for EV charging sessions. The model will be integrated into the EvoCharge Streamlit dashboard to provide users with real-time predictions.

### Key Applications

1. **Energy Consumption Prediction Model**

   - Target Variable: `energy_kWh`
   - Features: duration_min, session_type, session_day, time of day (extracted from start_time), user patterns, station patterns
   - Model Type: Random Forest Regressor or Gradient Boosting
   - Purpose: Predict how much energy a charging session will consume based on user inputs

2. **Feature Engineering**

   - Extract temporal features: hour of day, day of week, time of day categories (morning, afternoon, evening, night)
   - Create user behavior patterns: average consumption per user, charging frequency
   - Develop station characteristics: average energy per station, typical session duration
   - Calculate derived metrics: energy per minute, session efficiency

3. **Pattern Analysis**

   - Identify charging behavior differences between weekdays and weekends
   - Analyze energy consumption patterns across different session types
   - Understand temporal trends in charging behavior
   - Examine user and vehicle-specific consumption patterns

4. **Cost Estimation Foundation**
   - Use predicted energy consumption to calculate estimated charging costs
   - Combine with station characteristics from the charging stations dataset
   - Apply pricing models based on charger type, network, and location

### Integration with Charging Stations Dataset

This dataset will be used in conjunction with the EV Charging Stations dataset to:

- Provide context for station-specific predictions
- Enable location-based cost estimation
- Support charger type selection (Level 2 vs DC Fast)
- Enhance prediction accuracy with station characteristics

### Dashboard Implementation

The trained model will power an interactive prediction tool where users can:

- Input their desired charging duration
- Select session type and timing
- Choose a location and station type
- Receive predictions for energy consumption and estimated cost
- Compare scenarios across different times, locations, and charger types

## Data Statistics

- **Duration Range**: 30-120 minutes (mean: 75.3 min, std: 26.4 min)
- **Energy Range**: 7.89-113.36 kWh (mean: 41.9 kWh, std: 20.2 kWh)
- **Session Types**: Regular, Occasional, Emergency
- **Day Types**: Weekday, Weekend
- **Unique Users**: Multiple users with varying charging patterns
- **Unique Stations**: Multiple stations with different characteristics
- **Unique Vehicles**: Multiple vehicles with different consumption profiles

## Files in This Directory

- `charging_sessions.ipynb` - Jupyter notebook for data loading and exploration
- `ev_charging_sessions.csv` - Main dataset file with all charging session records
- `README.MD` - This documentation file

## Next Steps

1. Exploratory Data Analysis (EDA)

   - Visualize distributions of energy consumption and duration
   - Analyze correlations between features
   - Identify patterns and outliers
   - Examine temporal trends

2. Feature Engineering

   - Create time-based features
   - Develop user and station aggregates
   - Engineer interaction features
   - Handle categorical variables

3. Model Development

   - Train/test split with temporal considerations
   - Build baseline models
   - Implement Random Forest and XGBoost models
   - Perform hyperparameter tuning
   - Evaluate model performance (RMSE, MAE, R²)

4. Model Deployment
   - Export trained model for Streamlit integration
   - Create prediction pipeline
   - Implement real-time inference
   - Add model monitoring capabilities
