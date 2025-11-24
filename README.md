# EvoCharge - Smart EV Charging Predictor

## Project Overview

EvoCharge is an interactive data science project that predicts electric vehicle charging session energy consumption and estimates costs using machine learning. The project combines real-world EV charging station data with historical charging session patterns to provide users with actionable insights for planning their charging sessions.

## Problem Statement

Electric vehicle owners face uncertainty when planning charging sessions:

- How much energy will a charging session consume?
- What will the charging session cost?
- Which charging station and charger type should they use?
- When is the optimal time to charge to minimize costs?

EvoCharge addresses these questions by building a predictive model that estimates energy consumption based on session characteristics and provides cost estimates based on station attributes.

## Project Structure

```
EvoCharge/
├── streamlit_app.py                        # Main Streamlit dashboard application
├── app/                                    # Dashboard documentation and models
│   ├── README.md                          # Dashboard documentation
│   └── models/                            # Trained ML models (to be created)
├── data/                                   # Data collection and processing
│   ├── ev_charging_sessions/              # Charging session dataset
│   │   ├── charging_sessions.ipynb        # Data loading and exploration
│   │   ├── ev_charging_sessions.csv       # Session data (3,500 records)
│   │   └── README.MD                      # Dataset documentation
│   ├── ev_charging_stations/              # Charging station dataset
│   │   ├── ev_charging_stations.ipynb     # Data loading, cleaning, and analysis
│   │   ├── ev_charging_stations_feb2024_cleaned.csv
│   │   ├── ev_charging_stations_feb2024_public.csv
│   │   ├── ev_charging_stations_jan2023_cleaned.csv
│   │   └── README.md                      # Dataset documentation
│   └── afdc/                              # Alternative Fuel Data Center data
│       ├── data.ipynb
│       ├── afdc_stations_raw.csv
│       └── afdc_stations_top50.csv
├── requirements.txt                        # Python dependencies
├── env.example                            # Environment variables template
└── README.md                              # This file
```

## Datasets

### 1. EV Charging Sessions Dataset

- **Source**: Kaggle - Electric Vehicle Charging Sessions Dataset
- **Records**: 3,500 charging sessions
- **Features**: session_id, user_id, vehicle_id, station_id, start_time, end_time, duration_min, energy_kWh, session_day, session_type
- **Purpose**: Training data for machine learning model to predict energy consumption
- **Location**: `data/ev_charging_sessions/`

### 2. EV Charging Stations Dataset

- **Source**: Kaggle - EV Charging Stations US
- **Records**: 65,134 stations (February 2024), 54,238 stations (January 2023)
- **Features**: Station Name, Address, City, State, ZIP, charger types (Level 1/2/DC Fast), network, access type, facility type
- **Purpose**: Provide real-world station context for cost estimation and geographic analysis
- **Location**: `data/ev_charging_stations/`

## Methodology

### Data Processing

1. **Data Collection**

   - Downloaded EV charging sessions data from Kaggle
   - Downloaded EV charging stations data from Kaggle (two time periods)

2. **Data Cleaning**

   - Filled NaN values in charger count columns with 0 (indicating absence of that charger type)
   - Created derived features: Total_Chargers, Has_Level1, Has_Level2, Has_DC_Fast
   - Filtered public vs private stations
   - Standardized column names and data types

3. **Exploratory Data Analysis**
   - Analyzed distribution of charging session durations and energy consumption
   - Examined charger type availability across stations
   - Studied geographic distribution of charging infrastructure
   - Identified network provider and facility type patterns

### Machine Learning Model

**Objective**: Predict energy consumption (kWh) for a charging session

**Target Variable**: `energy_kWh`

**Features**:

- Duration of charging session (minutes)
- Session type (Regular, Occasional, Emergency)
- Day type (Weekday vs Weekend)
- Time of day (extracted from start_time)
- User charging patterns
- Station characteristics

**Model Type**: Random Forest Regressor or Gradient Boosting

**Evaluation Metrics**: RMSE, MAE, R-squared

### Cost Estimation Model

**Pricing Rules** (based on industry research):

Base Pricing by Charger Type:

- DC Fast Charging: $0.40-0.60 per kWh
- Level 2 Charging: $0.20-0.30 per kWh
- Level 1 Charging: $0.10-0.15 per kWh

Pricing Modifiers:

- Network provider (Shell Recharge +10%, Non-Networked baseline)
- Access type (public +15%, private baseline)
- Geographic location (urban/high-cost areas +20%)
- Facility type (parking garages +25%, utilities baseline)
- Time of day (peak hours +20%, off-peak -10%)

**Cost Calculation**:

```
Estimated Cost = Predicted Energy (kWh) × Base Price × Modifiers
```

## Dashboard Features

The Streamlit dashboard provides an interactive interface for users to:

1. **Select Location**: Choose geographic area using interactive map
2. **Choose Station**: Select from nearby charging stations with detailed characteristics
3. **Configure Session**: Set charging duration, time, session type, and charger type
4. **View Predictions**: See predicted energy consumption and estimated cost
5. **Compare Scenarios**: Analyze different charging options by time, location, and charger type
6. **Get Recommendations**: Receive smart suggestions for cost savings and optimal charging times

## Key Insights

### Charging Sessions Data

- Average session duration: 75.3 minutes (range: 30-120 min)
- Average energy consumption: 41.9 kWh (range: 7.89-113.36 kWh)
- Session types: Regular, Occasional, Emergency
- Temporal patterns: Weekday vs Weekend charging behavior

### Charging Stations Data

- Total stations: 65,134 (February 2024)
- Public stations: 61,308 (94.1%)
- Private stations: 3,826 (5.9%)
- Charger distribution:
  - Level 2: 56,677 stations (most common)
  - DC Fast: 9,364 stations
  - Level 1: 667 stations (rare)
- Top networks: ChargePoint (33,918), Non-Networked (9,180), Blink (5,528)
- Top states: California (16,455), New York (3,975), Florida (3,433)

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone or download the project repository

2. Navigate to the project directory:

```bash
cd EvoCharge
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):

```bash
cp env.example .env
# Edit .env with your configuration
```

### Running the Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Dependencies

Key Python packages:

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scikit-learn: Machine learning models
- xgboost: Gradient boosting (optional)
- streamlit: Web dashboard framework
- plotly: Interactive visualizations
- folium: Interactive maps
- kagglehub: Kaggle dataset downloads
- openpyxl: Excel file reading

See `requirements.txt` for complete list with versions.

## Project Workflow

1. **Data Collection**: Download datasets from Kaggle
2. **Data Cleaning**: Process and clean raw data (see notebooks)
3. **Exploratory Analysis**: Understand patterns and relationships in data
4. **Feature Engineering**: Create derived features for modeling
5. **Model Training**: Build and train predictive model (to be implemented)
6. **Model Evaluation**: Assess model performance and tune hyperparameters
7. **Dashboard Development**: Create interactive Streamlit application
8. **Deployment**: Deploy dashboard for user access

## Future Enhancements

### Model Improvements

- Incorporate weather data for seasonal patterns
- Add vehicle-specific consumption profiles
- Implement time series forecasting for demand prediction
- Develop user preference learning system

### Dashboard Features

- User accounts for saving preferences and history
- Real-time station availability integration
- Mobile-responsive design
- Push notifications for optimal charging times
- Social features for sharing cost-saving tips

### Data Expansion

- Integrate additional data sources
- Expand geographic coverage
- Add real-time pricing data
- Include vehicle API integration

## Use Cases

### Daily Commuter

A user who charges regularly at work can use EvoCharge to:

- Predict daily charging costs
- Identify optimal charging times to save money
- Compare nearby station options

### Road Trip Planner

A user planning a long-distance trip can:

- Find charging stations along their route
- Estimate total charging costs for the trip
- Optimize charging stops for cost and time

### Cost-Conscious User

A user focused on minimizing expenses can:

- Identify cheapest charging times and locations
- Compare Level 2 vs DC Fast charging costs
- Track monthly charging expenses

### Fleet Manager

A business managing multiple EVs can:

- Optimize charging schedules for multiple vehicles
- Analyze cost patterns across locations
- Plan infrastructure investments

## Technical Details

### Data Pipeline

```
Raw Data → Data Cleaning → Feature Engineering → Model Training → Prediction
                                                                      ↓
Station Data → Pricing Rules → Cost Calculation → Dashboard Display
```

### Model Architecture

- Input: Session characteristics (duration, type, time, etc.)
- Processing: Feature engineering and transformation
- Model: Ensemble learning (Random Forest/XGBoost)
- Output: Energy consumption prediction (kWh)
- Post-processing: Cost estimation using station attributes

## Contributing

This project is part of a data science portfolio. Suggestions and feedback are welcome.

## Data Sources and Credits

- **EV Charging Sessions Dataset**: Kaggle user zyan1999
- **EV Charging Stations Dataset**: Kaggle user salvatoresaia

## License

This project is for educational and portfolio purposes.

## Contact

For questions or feedback about this project, please refer to the project repository.

## Acknowledgments

Special thanks to the Kaggle community and the U.S. Department of Energy for providing publicly available EV charging data that made this project possible.
