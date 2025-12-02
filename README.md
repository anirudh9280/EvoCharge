# EvoCharge - California EV Charging Cost Predictor

## Project Overview

EvoCharge is an interactive Streamlit dashboard that predicts electric vehicle charging costs for California using machine learning. The project combines 16,455 California charging stations with county-specific electricity rates and a Lasso regression model trained on 3,500 charging sessions to provide real-time cost estimates for EV owners.

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
├── streamlit_app.py                        # Main Streamlit dashboard (✓ IMPLEMENTED)
├── models-shohom.py                        # Lasso model training script
├── DESIGN.md                              # System architecture documentation
├── data/                                   # Data collection and processing
│   ├── ev_charging_sessions/              # Charging session dataset
│   │   ├── charging_sessions.ipynb        # Data loading and exploration
│   │   ├── ev_charging_sessions.csv       # Session data (3,500 records)
│   │   ├── model_training_*.ipynb         # Model training notebooks (Brandon & Shohom)
│   │   ├── EDA_charging_sessions.ipynb    # Exploratory data analysis
│   │   └── README.MD                      # Dataset documentation
│   ├── ev_charging_stations/              # Charging station dataset
│   │   ├── ev_charging_stations.ipynb     # Data loading, cleaning, and analysis
│   │   ├── ev_charging_stations_feb2024_cleaned.csv  # 16,455 CA stations
│   │   └── README.md                      # Dataset documentation
│   ├── ca_county_prices/                  # California county electricity rates (✓ NEW)
│   │   ├── ev_charging_stations_county_prices.csv  # Stations + rates
│   │   ├── ca_county_rates.csv            # 58 counties with rates
│   │   ├── ca_zip_to_county.csv           # 2,447 ZIP to county mappings
│   │   └── scrape_ca_rates.ipynb          # Rate collection notebook
│   └── afdc/                              # Alternative Fuel Data Center data
│       ├── data.ipynb
│       ├── afdc_stations_raw.csv
│       └── afdc_stations_top50.csv
├── requirements.txt                        # Python dependencies
└── README.md                              # This file
```

## Datasets

### 1. EV Charging Sessions Dataset

- **Source**: Kaggle - Electric Vehicle Charging Sessions Dataset
- **Records**: 3,500 charging sessions
- **Features**: session_id, user_id, vehicle_id, station_id, start_time, end_time, duration_min, energy_kWh, session_day, session_type
- **Purpose**: Training data for Lasso regression model to predict charging power
- **Location**: `data/ev_charging_sessions/`

### 2. California EV Charging Stations Dataset

- **Source**: Kaggle - EV Charging Stations US (filtered for California)
- **Records**: 16,455 California charging stations
- **Features**: Station Name, Address, City, State, ZIP, charger types (Level 1/2/DC Fast), network, access type, facility type
- **Purpose**: Real-world station data for interactive map and location-based predictions
- **Location**: `data/ev_charging_stations/`

### 3. California County Electricity Rates

- **Source**: California utility rate data + ZIP-to-county crosswalk files
- **Records**: 58 California counties with electricity rates
- **Rate Range**: $0.30/kWh (cheapest) to $0.4597/kWh (standard)
- **Purpose**: County-specific cost calculations for accurate price predictions
- **Location**: `data/ca_county_prices/`
- **Key Files**:
  - `ca_county_rates.csv`: Rate per kWh for each county
  - `ca_zip_to_county.csv`: 2,447 ZIP codes mapped to counties
  - `ev_charging_stations_county_prices.csv`: Stations with merged rate data

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

**Model**: Lasso Regression (L1-regularized linear model)

**Author**: Shohom's implementation from `shohom-models` branch

**Approach**: Predicts average charging power (kW), then multiplies by duration to get energy (kWh)

- More physically sound than directly predicting energy
- Linear relationship is interpretable
- L1 regularization prevents overfitting

**Features**:

- `duration_hours`: Charging session duration in hours (MAJOR impact)
- `start_hour`: Start time as decimal hour (MEDIUM impact)
- `session_day`: Weekday vs Weekend (MINOR impact)
- `session_type`: Regular, Occasional, or Emergency (MINOR impact ~3%)

**Performance Metrics** (on test set):

- **MAE**: ~11-12 kWh (Mean Absolute Error)
- **RMSE**: ~14-15 kWh (Root Mean Squared Error)
- **R² Score**: ~0.40-0.50 (explains 40-50% of variance)
- **Improvement**: ~30-40% better than baseline (simple mean prediction)

**Training Details**:

- Training samples: 2,800 sessions (80%)
- Testing samples: 700 sessions (20%)
- Cross-validation: 5-fold CV for alpha selection
- Preprocessing: StandardScaler for numeric, OneHotEncoder for categorical

### Cost Calculation

**County-Based Electricity Rates**:

California counties are organized into three rate tiers:

- **Lowest Tier ($0.30/kWh)**: Los Angeles, Imperial, Alpine, Del Norte, Siskiyou, Modoc (6 counties)
- **Low Tier ($0.3869/kWh)**: Orange, Riverside, San Bernardino, Ventura, Sacramento, Inyo, Mono (7 counties)
- **Standard Tier ($0.4597/kWh)**: San Francisco, San Diego, Santa Clara, and 42 other counties

**Cost Calculation Formula**:

```python
# Step 1: Model predicts average charging power
avg_power_kw = lasso_model.predict(session_features)

# Step 2: Calculate energy
energy_kwh = avg_power_kw × duration_hours

# Step 3: Calculate cost using county rate
total_cost = energy_kwh × county_electricity_rate

# Example: 60-minute charge in Los Angeles
# Power: 35 kW → Energy: 35 kWh → Cost: 35 × $0.30 = $10.50
```

**Rate Distribution**:

- 2,206 stations at $0.30/kWh (23% - cheapest)
- 2,064 stations at $0.3869/kWh (22%)
- 5,258 stations at $0.4597/kWh (55%)

**Top Counties by Station Count**:

- Los Angeles: 2,175 stations ($0.30/kWh - cheapest!)
- Orange: 1,109 stations ($0.3869/kWh)
- Santa Clara: 1,100 stations ($0.4597/kWh)
- San Mateo: 749 stations ($0.4597/kWh)
- San Diego: 680 stations ($0.4597/kWh)

## Dashboard Features

The Streamlit dashboard (`streamlit_app.py`) provides a complete interactive interface:

### 1. **ZIP Code Location Input** ✓

- Enter any California ZIP code
- Automatically identifies county
- Displays county-specific electricity rate
- Filters stations to your area

### 2. **Charging Session Configuration** ✓

- **Duration slider**: 30-120 minutes
- **Start time picker**: Select when you'll charge (decimal hour)
- **Day type**: Weekday or Weekend
- **Session type**: Regular, Occasional, or Emergency

### 3. **Real-Time Cost Prediction** ✓

Displays 4 key metrics:

- **Predicted Energy**: Total kWh consumption
- **Avg Charging Power**: kW rate (model output)
- **Estimated Cost**: Total price in USD
- **Cost per Minute**: $/min rate

### 4. **Interactive California Map** ✓

- **16,455 stations** plotted on Folium map
- **Color-coded by rate**: 🟢 Green (cheap), 🔵 Blue (mid), 🟠 Orange (standard)
- **Station details on hover**: Name, address, network, charger counts
- **Auto-zoom**: Centers on your county when ZIP entered
- **Performance optimized**: Shows 75-100 stations, full table below
- **Pre-cached coordinates**: Instant loading (no slow geocoding)

### 5. **Complete Station Browser** ✓

- Searchable, sortable table of all filtered stations
- Shows: Name, Address, City, County, Rate, Charger counts
- Filter options: DC Fast only, minimum chargers
- Search by station name, city, or address

### 6. **Model Performance Metrics** ✓

Comprehensive model transparency with:

- **MAE, RMSE, R² Score**: Test set performance
- **Baseline comparison**: Shows % improvement
- **Training details**: Sample sizes, selected alpha
- **Interpretation guide**: Explains what metrics mean

### 7. **Cost Breakdown & Insights** ✓

- Detailed calculation explanation
- County rate comparison (top 10 cheapest)
- Network distribution statistics
- Physical reasoning: Energy = Power × Time

### 8. **Sidebar Information** ✓

- Model performance summary
- Feature importance explanation
- Why Lasso regression was chosen
- Data sources and credits

## Key Insights & Findings

### Model Insights

**Session Type Impact** (Historical Averages):

- Regular: 42.07 kWh average
- Occasional: 41.14 kWh average (-2.2%)
- Emergency: 42.36 kWh average (+0.7%)
- **Key Finding**: Session type has minimal impact (~3% variation) - duration is the primary driver

**Model Performance**:

- Lasso regression significantly outperforms baseline
- 30-40% improvement in prediction accuracy
- Physically sound approach (power × time)
- Interpretable linear relationships

### California Charging Infrastructure

**Station Distribution**:

- **Total CA stations**: 16,455 (25% of US total)
- **Geographic concentration**:
  - Top 3 counties: 46% of all CA stations
  - Los Angeles dominates: 2,175 stations (13%)

**Electricity Rate Insights**:

- **53% price difference** between cheapest and most expensive counties
- Los Angeles ($0.30/kWh) vs. Most counties ($0.4597/kWh)
- Example cost difference (60-min charge, 35 kWh):
  - Los Angeles: $10.50
  - San Francisco: $16.09
  - **Savings**: $5.59 (35% cheaper in LA!)

**Charger Type Distribution** (California):

- Level 2 chargers: Most common (standard charging)
- DC Fast chargers: Growing (quick charging)
- Level 1 chargers: Rare (slow charging)

### Cost-Saving Opportunities

**Location matters most**:

- Charging in Los Angeles County saves ~35% vs San Francisco
- 6 counties offer the lowest rates ($0.30/kWh)
- 45 counties at standard rates ($0.4597/kWh)

**Duration is the primary cost driver**:

- Linear relationship: 2× duration ≈ 2× cost
- Session type has minimal impact
- Time of day: Minor effect in current model

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

### Core Libraries:

- **pandas** (>=1.5.0): Data manipulation and analysis
- **numpy** (>=1.21.0): Numerical computing
- **scikit-learn** (>=1.1.0): Lasso model, preprocessing, metrics

### Web Framework:

- **streamlit** (>=1.25.0): Dashboard application framework
- **streamlit-folium** (>=0.13.0): Folium map integration

### Visualization:

- **folium** (>=0.14.0): Interactive California map
- **plotly** (>=5.10.0): Interactive charts
- **matplotlib** (>=3.5.0): Static plots
- **seaborn** (>=0.11.0): Statistical visualizations

### Data Collection:

- **kagglehub**: Kaggle dataset downloads
- **openpyxl** (>=3.0.0): Excel file reading
- **requests** (>=2.28.0): API calls

### Geospatial (Optional):

- **pydeck** (>=0.8.0): Advanced 3D mapping
- **geopandas** (>=0.12.0): Geospatial analysis

See `requirements.txt` for complete list with exact versions.

## Project Workflow (Completed)

1. **Data Collection** ✓

   - Downloaded EV charging sessions (3,500 records)
   - Downloaded US charging stations, filtered to California (16,455 stations)
   - Collected California county electricity rates (58 counties)
   - Built ZIP-to-county mapping (2,447 ZIP codes)

2. **Data Cleaning & Integration** ✓

   - Processed station data, handled missing values
   - Mapped ZIP codes to counties using HUD crosswalk files
   - Merged electricity rates with station data
   - 9,528 stations successfully mapped with rates (58% coverage)

3. **Exploratory Data Analysis** ✓

   - Analyzed session duration vs energy consumption
   - Examined session type distributions and impacts
   - Studied California charging infrastructure
   - Identified county rate variations

4. **Feature Engineering** ✓

   - Created `duration_hours` from timestamps
   - Extracted `start_hour` as decimal hour
   - Engineered station aggregates by city
   - Pre-cached city coordinates for mapping

5. **Model Development** ✓

   - **Team collaboration**: Multiple models developed
     - Brandon: Random Forest approaches
     - Shohom: Lasso, Random Forest, SVR
   - **Final selection**: Shohom's Lasso model (best interpretability)
   - Training: 80/20 split, 5-fold CV for alpha selection
   - Evaluation: MAE, RMSE, R², baseline comparison

6. **Dashboard Development** ✓

   - Built complete Streamlit application
   - Integrated Lasso model with real-time predictions
   - Created interactive Folium map of California stations
   - Implemented county-based cost calculation
   - Added comprehensive model metrics display

7. **Performance Optimization** ✓

   - Pre-cached city coordinates (instant loading)
   - Optimized map rendering (100 stations max)
   - Fixed map flickering issues
   - Implemented efficient caching strategies

8. **Documentation** ✓
   - Comprehensive README (this file)
   - System design documentation (DESIGN.md)
   - Dataset documentation in each data folder
   - Inline code comments throughout

## Future Enhancements

### Model Improvements

- [ ] Incorporate weather/temperature data for seasonal adjustments
- [ ] Add vehicle-specific consumption profiles (by make/model)
- [ ] Implement time-of-use (TOU) rate variations
- [ ] Develop ensemble model combining Lasso + Random Forest
- [ ] Add confidence intervals to predictions

### Dashboard Features

- [ ] User accounts for saving favorite stations and history
- [ ] Real-time station availability integration via APIs
- [ ] Mobile-responsive design optimization
- [ ] Route planning: Multi-station trip cost estimation
- [ ] Push notifications for optimal charging times
- [ ] Historical cost tracking and monthly summaries

### Data Expansion

- [ ] Improve ZIP-to-county mapping coverage (currently 58%)
- [ ] Expand to other states beyond California
- [ ] Integrate real-time utility rate APIs
- [ ] Add actual station coordinates (vs city-level geocoding)
- [ ] Include vehicle API integration for personalized predictions
- [ ] Incorporate demand-based dynamic pricing

### Performance Optimizations

- [ ] Implement progressive loading for map markers
- [ ] Add clustering for high-density areas
- [ ] Cache model predictions for common scenarios
- [ ] Build separate prediction API service
- [ ] Add database backend for faster queries

## Use Cases

### Example 1: Daily Commuter in Los Angeles

**Scenario**: Charges during lunch break at work

- **Input**: ZIP 90012, 60 minutes, 12:00 PM, Weekday, Regular
- **Prediction**: ~35 kWh, ~7 kW avg power
- **County Rate**: $0.30/kWh (Los Angeles - cheapest in CA!)
- **Cost**: **$10.50**
- **Benefit**: Saves $5.59 vs charging in San Francisco

### Example 2: Weekend Shopper in San Francisco

**Scenario**: Charges while shopping on Saturday afternoon

- **Input**: ZIP 94105, 90 minutes, 2:00 PM, Weekend, Occasional
- **Prediction**: ~50 kWh, ~33 kW avg power
- **County Rate**: $0.4597/kWh (San Francisco - standard tier)
- **Cost**: **$22.99**
- **Map**: Shows 100+ nearby stations, filter by DC Fast

### Example 3: Road Trip Planner

**Scenario**: Planning route from San Diego to San Francisco

- **Use dashboard to**:
  - Check rates in each county along route
  - Find stations in different cities
  - Estimate total trip charging cost
  - Identify cheapest charging locations

### Example 4: Cost-Conscious EV Owner

**Scenario**: Comparing charging locations to minimize costs

- **Discovery**: Los Angeles County is 35% cheaper than most counties
- **Action**: Plans trips to include charging in low-rate counties
- **Savings**: $5-6 per session adds up to $100-200/year

### Example 5: Data Science Student

**Scenario**: Understanding ML model performance

- **Explores**:
  - Model metrics (MAE, RMSE, R²)
  - Feature importance (duration > time > session type)
  - Baseline comparison (30-40% improvement)
  - Physical reasoning (power × time = energy)

## Technical Details

### Data Pipeline (Implemented)

```
┌─ Charging Sessions (3,500) ─┐
│  Kaggle Dataset              │
│  • duration, start_time      │──┐
│  • session_day, type         │  │
│  • energy_kWh (target)       │  │
└──────────────────────────────┘  │
                                  │
┌─ CA Stations (16,455) ───────┐  │
│  Filtered from 65K US         │  │
│  • Location, ZIP, City        │──┼──→ Feature Engineering
│  • Charger types, Network    │  │    • duration_hours
└──────────────────────────────┘  │    • start_hour (decimal)
                                  │    • OneHotEncode categories
┌─ County Rates (58 counties) ─┐  │
│  CA Utility Data              │  │
│  • county_name                │──┘
│  • rate_per_kwh ($0.30-0.46) │
└──────────────────────────────┘
         │                                    │
         │                                    ↓
         │                          ┌─────────────────┐
         │                          │ Lasso Model     │
         │                          │ (LassoCV)       │
         │                          ├─────────────────┤
         │                          │ StandardScaler  │
         │                          │ OneHotEncoder   │
         │                          └─────────────────┘
         │                                    │
         │                                    ↓
         │                          Predicts: avg_power (kW)
         │                                    │
         │                                    ↓
         │                          energy = power × duration
         │                                    │
         └────────────────────────────────────┼───→ Cost Calculation
                                              │     cost = energy × rate
                                              ↓
                                    ┌───────────────────┐
                                    │ Streamlit Display │
                                    │ • Predictions     │
                                    │ • Map             │
                                    │ • Metrics         │
                                    └───────────────────┘
```

### Model Architecture (Shohom's Lasso)

**Pipeline Structure**:

```python
Pipeline([
    ('preprocess', ColumnTransformer([
        ('num', StandardScaler(), ['duration_hours', 'start_hour']),
        ('cat', OneHotEncoder(), ['session_day', 'session_type'])
    ])),
    ('model', LassoCV(alphas=np.logspace(-3, 1, 20), cv=5))
])
```

**Flow**:

1. **Input**: Session features (duration, start_hour, day, type)
2. **Preprocessing**: Scale numeric, encode categorical
3. **Model**: Lasso predicts average power (kW)
4. **Calculation**: Energy (kWh) = Power (kW) × Duration (hours)
5. **Pricing**: Cost ($) = Energy (kWh) × County Rate ($/kWh)
6. **Output**: Energy prediction + cost estimate + metrics

## Contributing

This project is part of a data science portfolio. Suggestions and feedback are welcome.

## Data Sources and Credits

### Primary Data Sources

- **EV Charging Sessions Dataset**: Kaggle user zyan1999
- **EV Charging Stations Dataset**: Kaggle user salvatoresaia (filtered to CA)
- **California County Electricity Rates**: California utility rate data
- **ZIP-to-County Mapping**: HUD USPS ZIP-County crosswalk files

### Model Development

- **Lasso Model**: Shohom (selected for production)
- **Random Forest Models**: Brandon Van Horn
- **Dashboard Implementation**: Team collaboration
- **County Rate Integration**: Data pipeline development

### Tools & Libraries

- **scikit-learn**: Machine learning framework
- **Streamlit**: Dashboard framework
- **Folium**: Interactive mapping
- **pandas/numpy**: Data processing

## License

This project is for educational and portfolio purposes.

## Contact

For questions or feedback about this project, please refer to the project repository.

## Project Status

✅ **Completed** (December 2025)

### Deliverables

- ✓ Fully functional Streamlit dashboard
- ✓ Lasso regression model (MAE ~11-12 kWh)
- ✓ 16,455 California stations with county rates
- ✓ Interactive map with 100+ markers
- ✓ Comprehensive model metrics display
- ✓ ZIP code to cost prediction pipeline
- ✓ Complete documentation

### Performance

- **Model**: 30-40% better than baseline
- **Load Time**: < 1 second (optimized caching)
- **Map Rendering**: Instant (pre-cached coordinates)
- **User Experience**: Smooth, professional, responsive

## Acknowledgments

Special thanks to:

- **Kaggle community** for providing EV charging datasets
- **U.S. Department of Energy** for public charging station data
- **California utilities** for county rate information
- **Open source community** for scikit-learn, Streamlit, Folium, and other tools

---

**Note**: This project demonstrates end-to-end data science workflow including data collection, cleaning, feature engineering, model training, evaluation, deployment, and documentation.
