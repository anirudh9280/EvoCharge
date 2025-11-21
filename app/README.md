# EvoCharge Streamlit Dashboard

## Overview

EvoCharge is an interactive web application that helps electric vehicle owners make informed decisions about their charging sessions by predicting energy consumption and estimating costs. The dashboard combines machine learning predictions with real-world charging station data to provide personalized insights and recommendations.

## Purpose and Value Proposition

### What Problem Does This Solve?

EV owners face uncertainty when planning charging sessions:

- How much energy will I need?
- How much will it cost?
- Which charging station should I use?
- When is the best time to charge?
- Should I use Level 2 or DC Fast charging?

EvoCharge answers these questions by providing data-driven predictions and cost estimates before users commit to a charging session.

### Who Is This For?

- **EV Owners**: Planning daily charging routines and managing costs
- **Fleet Managers**: Optimizing charging schedules for multiple vehicles
- **EV Shoppers**: Understanding charging costs before purchasing an EV
- **Station Operators**: Analyzing demand patterns and pricing strategies
- **Researchers**: Studying EV charging behavior and infrastructure utilization

## How It Works

### Data Sources and Model Integration

The dashboard integrates three key components:

#### 1. EV Charging Sessions Dataset (3,500 records)

**Purpose**: Train the machine learning model to predict energy consumption

**Key Features Used**:

- Historical charging duration patterns
- Session types (Regular, Occasional, Emergency)
- Temporal patterns (weekday vs weekend, time of day)
- User and vehicle charging behavior
- Station-specific consumption patterns

**Model Output**: Predicted energy consumption in kWh

#### 2. EV Charging Stations Dataset (65,134 stations)

**Purpose**: Provide real-world context and enable cost estimation

**Key Features Used**:

- Station locations (city, state, ZIP)
- Charger types (Level 1, Level 2, DC Fast)
- Network providers (Shell Recharge, EV Connect, etc.)
- Access types (public vs private)
- Facility types (utility, parking garage, workplace, etc.)

**Model Output**: Pricing estimates based on station characteristics

#### 3. Machine Learning Prediction Model

**Algorithm**: Random Forest Regressor or Gradient Boosting

**Training Process**:

- Features: duration, session type, day type, time of day, user patterns
- Target: energy consumption (kWh)
- Validation: Cross-validation with temporal splits
- Metrics: RMSE, MAE, R-squared

**Prediction Pipeline**:

```
User Input → Feature Engineering → ML Model → Energy Prediction (kWh)
                                                        ↓
Station Characteristics → Pricing Rules → Cost Estimation ($)
```

### Cost Estimation Model

The dashboard applies research-based pricing rules:

**Base Pricing by Charger Type**:

- DC Fast Charging: $0.40-0.60 per kWh (fastest, premium pricing)
- Level 2 Charging: $0.20-0.30 per kWh (standard, moderate pricing)
- Level 1 Charging: $0.10-0.15 per kWh (slowest, economy pricing)

**Pricing Modifiers**:

- Network Premium: Shell Recharge (+10%), Tesla Supercharger (+15%)
- Access Type: Public stations (+15% vs private)
- Location: Urban/high-cost areas (+20%)
- Facility Type: Parking garages (+25%), retail locations (+10%)
- Time of Day: Peak hours (+20%), off-peak hours (-10%)

**Final Cost Calculation**:

```
Estimated Cost = Predicted Energy (kWh) × Base Price × Modifiers
```

## User Experience Flow

### Step 1: Location Selection

**What Users See**:

- Interactive map of the United States
- Heat map overlay showing station density
- Search bar for city, state, or ZIP code

**What Users Do**:

- Click on the map or search for their location
- View nearby charging stations
- See station density in their area

**What They Get**:

- Filtered list of stations in their region
- Station count by charger type
- Network provider distribution

### Step 2: Station Selection

**What Users See**:

- List of nearby stations with key details
- Station names and addresses
- Available charger types at each location
- Network provider and facility type
- Estimated pricing tier (low/medium/high)

**What Users Do**:

- Select a specific station OR
- Choose station characteristics (charger type, network, facility)
- View station details and specifications

**What They Get**:

- Selected station information
- Available charger types
- Access type (public/private)
- Facility type context

### Step 3: Session Configuration

**What Users See**:

- Charger type selector (Level 2 vs DC Fast)
- Duration slider (30-120 minutes)
- Date and time picker
- Session type selector (Regular/Occasional/Emergency)

**What Users Do**:

- Choose their preferred charger type
- Set expected charging duration
- Select when they plan to charge
- Indicate session type based on their needs

**What They Get**:

- Real-time preview of inputs
- Visual feedback on selections
- Validation of input ranges

### Step 4: Prediction and Results

**What Users See**:

- **Energy Prediction Card**
  - Predicted energy consumption in kWh
  - Confidence interval or range
  - Comparison to average sessions
- **Cost Estimation Card**
  - Estimated total cost in dollars
  - Cost breakdown by components
  - Price per kWh rate
- **Environmental Impact Card**
  - CO2 emissions saved vs gasoline
  - Equivalent trees planted
  - Environmental benefit metrics
- **Visual Gauges and Charts**
  - Animated battery fill gauge
  - Energy consumption visualization
  - Cost comparison bar chart

**What Users Do**:

- Review predictions and estimates
- Adjust inputs to explore scenarios
- Compare different options

**What They Get**:

- Actionable predictions for their charging session
- Cost transparency before charging
- Confidence in their charging decisions

### Step 5: Scenario Comparison

**What Users See**:

- **Compare by Time**: Cost and energy predictions across different times of day
- **Compare by Charger**: Level 2 vs DC Fast comparison (cost vs speed tradeoff)
- **Compare by Location**: Different stations in the same area
- Interactive charts and visualizations

**What Users Do**:

- Explore different charging scenarios
- Identify cost-saving opportunities
- Find optimal charging times and locations

**What They Get**:

- **Smart Recommendations**:
  - "Save $3.50 by charging 2 hours later"
  - "DC Fast costs 40% more but saves 45 minutes"
  - "Station X is 0.5 miles farther but $2.20 cheaper"
- **Insights and Alerts**:
  - Peak pricing warnings
  - High demand period notifications
  - Best time to charge suggestions
  - Station availability predictions

## Dashboard Features

### Interactive Components

1. **Geographic Visualization**

   - Folium or Plotly map with station markers
   - Heat map showing station density
   - Clickable markers with station details
   - Distance calculations from user location

2. **Prediction Interface**

   - Streamlit sliders for duration and time
   - Radio buttons for session type and charger type
   - Date/time picker for scheduling
   - Real-time prediction updates

3. **Results Display**

   - Metric cards with large, clear numbers
   - Plotly gauge charts for battery visualization
   - Animated number counters
   - Color-coded indicators (green/yellow/red)

4. **Comparison Tools**

   - Tabbed interface for different comparison types
   - Line charts for temporal comparisons
   - Bar charts for station comparisons
   - Scatter plots for pattern analysis

5. **Recommendation Engine**
   - Smart alerts based on user inputs
   - Cost-saving suggestions
   - Optimal timing recommendations
   - Station alternatives

### Technical Implementation

**Frontend (Streamlit)**:

- `streamlit` for web framework
- `plotly` for interactive visualizations
- `folium` for mapping
- `pandas` for data manipulation
- Custom CSS for styling and animations

**Backend (Python)**:

- `scikit-learn` or `xgboost` for ML model
- `joblib` or `pickle` for model serialization
- `numpy` for numerical computations
- Custom pricing engine for cost calculations

**Data Pipeline**:

```
CSV Files → Pandas DataFrames → Feature Engineering → Model Input
                                                            ↓
                                                    Model Prediction
                                                            ↓
Station Data → Pricing Rules → Cost Calculation → Dashboard Output
```

## Key Benefits for Users

### 1. Cost Transparency

Users know exactly how much their charging session will cost before they start, eliminating billing surprises.

### 2. Informed Decision Making

Compare scenarios to find the optimal balance between cost, speed, and convenience.

### 3. Time Savings

Avoid peak hours and congested stations by seeing predicted demand patterns.

### 4. Money Savings

Identify cheaper charging options and optimal times to charge at lower rates.

### 5. Environmental Awareness

Understand the environmental impact of their EV usage compared to gasoline vehicles.

### 6. Planning Confidence

Make better decisions about where and when to charge based on data-driven predictions.

## Example Use Cases

### Use Case 1: Daily Commuter

**Scenario**: Sarah needs to charge her EV after work

**User Journey**:

1. Opens EvoCharge dashboard
2. Searches for "Los Angeles, CA"
3. Sees 1,247 nearby stations
4. Selects Level 2 charger at workplace parking garage
5. Sets duration to 60 minutes, time to 5:00 PM
6. Gets prediction: 42 kWh, $13.50
7. Sees recommendation: "Charge at 7:00 PM to save $2.70"
8. Adjusts time and confirms lower cost

**Value Gained**: Saved $2.70 by charging during off-peak hours

### Use Case 2: Road Trip Planner

**Scenario**: Mike is planning a long-distance trip

**User Journey**:

1. Opens EvoCharge dashboard
2. Searches multiple cities along route
3. Compares DC Fast stations at each stop
4. Sees costs: Stop 1 ($22), Stop 2 ($18), Stop 3 ($25)
5. Identifies Stop 2 as optimal (cheaper, good location)
6. Plans charging stops accordingly

**Value Gained**: Optimized route for cost and convenience

### Use Case 3: Cost-Conscious User

**Scenario**: Lisa wants to minimize charging costs

**User Journey**:

1. Opens EvoCharge dashboard
2. Selects her usual station
3. Uses comparison tool to check different times
4. Sees cost range: $8 (2:00 AM) to $15 (6:00 PM)
5. Identifies pattern: overnight charging is cheapest
6. Adjusts charging schedule

**Value Gained**: Saves ~$140/month by charging overnight

## Future Enhancements

### Planned Features

- User accounts to save preferences and history
- Real-time station availability integration
- Mobile app version
- Push notifications for optimal charging times
- Integration with vehicle APIs for automatic recommendations
- Social features to share cost-saving tips
- Historical tracking of user's charging costs
- Predictive maintenance alerts for stations

### Advanced Analytics

- Seasonal pricing predictions
- Demand forecasting for specific stations
- Network-wide optimization recommendations
- Fleet management dashboard for businesses

## Technical Requirements

### Dependencies

See `requirements.txt` for complete list:

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- plotly >= 5.17.0
- folium >= 0.14.0
- joblib >= 1.3.0

### Running the Dashboard

```bash
cd app/
streamlit run streamlit_app.py
```

### Environment Variables

See `env.example` for configuration options

## Project Structure

```
EvoCharge/
├── app/
│   ├── streamlit_app.py          # Main dashboard application
│   ├── README.md                  # This file
│   └── models/                    # Trained ML models (to be created)
│       └── energy_predictor.pkl
├── data/
│   ├── ev_charging_sessions/      # Session data for ML training
│   │   ├── ev_charging_sessions.csv
│   │   └── README.MD
│   └── ev_charging_stations/      # Station data for context
│       ├── ev_charging_stations.csv (to be created)
│       └── README.md
└── requirements.txt
```

## Conclusion

EvoCharge transforms the EV charging experience by providing transparent, data-driven predictions that help users make better decisions. By combining machine learning with real-world station data, the dashboard delivers actionable insights that save users time and money while promoting efficient use of charging infrastructure.

The dashboard bridges the gap between uncertainty and confidence, turning the question "How much will this cost?" into a clear, reliable answer that users can trust.
