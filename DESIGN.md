flowchart LR
  A["📊 Data<br/>Sessions<br/>Stations<br/>Rates"]
  B["🔧 Process<br/>Features<br/>ZIP→County"]
  C["🤖 Lasso Model<br/>Predict Energy"]
  D["💰 Cost<br/>Energy × Rate"]
  E["📱 Dashboard<br/>Map + Results"]

  A --> B --> C --> D --> E

  style A fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
  style B fill:#fff3e0,stroke:#f57c00,stroke-width:2px
  style C fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
  style D fill:#fff9c4,stroke:#f9a825,stroke-width:2px
  style E fill:#e8f5e9,stroke:#388e3c,stroke-width:3px


  %% =========================
  %% Data Sources
  %% =========================
  A1[Kaggle API EV Charging Sessions 3,500 records]
  A2[Kaggle API EV Charging Stations US 65K+ stations Feb 2024 54K+ stations Jan 2023]
  A3[AFDC Data Alternative Fuels Data Center Top 50 stations]
  A4[Industry Research Pricing Models Charger Specifications]

  %% =========================
  %% ETL + Raw Data
  %% =========================
  B0[(EvoCharge ETL Pipeline)]
  B1[kagglehub.dataset_download fetch sessions data auto-caching]
  B2[kagglehub.dataset_download fetch stations data Excel to DataFrame]
  B3[manual collection AFDC scraping CSV export]
  B4[pricing_rules.py define cost models modifier logic]

  %% =========================
  %% Storage - Raw Data
  %% =========================
  C1[(ev_charging_sessions/ ev_charging_sessions.csv)]
  C2[(ev_charging_stations/ feb2024_cleaned.csv jan2023_cleaned.csv feb2024_public.csv)]
  C3[(afdc/ afdc_stations_raw.csv afdc_stations_top50.csv)]

  %% =========================
  %% Data Cleaning
  %% =========================
  D0[Data Cleaning Layer]
  D1[charging_sessions.ipynb load + validate no missing values]
  D2[ev_charging_stations.ipynb fill NaN with 0 create derived features]
  D3[data.ipynb AFDC processing geographic filtering]

  %% =========================
  %% Feature Engineering
  %% =========================
  E0[Feature Engineering]
  E1[Temporal Features hour of day day of week weekday vs weekend]
  E2[Station Features Total_Chargers Has_Level1/2/DC_Fast charger type flags]
  E3[User Patterns avg consumption per user charging frequency station preferences]
  E4[Session Features energy per minute session efficiency type encoding]
  E5[(features_engineered.csv model-ready dataset)]

  %% =========================
  %% Modeling Pipeline
  %% =========================
  F0[ML Model Development]
  F1[energy_prediction_model.ipynb train/test split temporal validation]
  F2[Models Random Forest Regressor Gradient Boosting XGBoost]
  F3[Hyperparameter Tuning GridSearchCV RandomizedSearchCV]
  F4[Model Evaluation RMSE / MAE / R² feature importance]
  F5[(models/ energy_predictor.pkl scaler.pkl feature_names.pkl)]

  %% =========================
  %% Pricing Engine
  %% =========================
  G0[Cost Estimation Engine]
  G1[Base Pricing DC Fast: $0.40-0.60/kWh Level 2: $0.20-0.30/kWh Level 1: $0.10-0.15/kWh]
  G2[Pricing Modifiers network premium access type location factor facility type time of day]
  G3[Cost Calculator Predicted kWh × Base × Modifiers]
  G4[(pricing_model.py cost estimation logic)]

  %% =========================
  %% Analytics Engine
  %% =========================
  H0((EvoCharge Analytics Engine))
  H1[Prediction Service energy consumption forecast confidence intervals]
  H2[Cost Estimator real-time pricing scenario comparison]
  H3[Station Analyzer filter by location charger availability network distribution]
  H4[Recommendation Engine optimal charging times cost-saving suggestions station alternatives]

  %% =========================
  %% Streamlit Dashboard
  %% =========================
  I0[Streamlit Dashboard streamlit_app.py]
  I1[Location Selector interactive map station density heatmap city/state search]
  I2[Station Browser filter by charger type network provider access type facility type]
  I3[Session Configurator duration slider time picker session type selector charger type choice]
  I4[Prediction Display energy consumption card cost estimation card CO2 savings card animated gauges]
  I5[Scenario Comparator compare by time compare by charger compare by location interactive charts]
  I6[Insights Panel smart recommendations peak pricing alerts demand predictions cost-saving tips]
  I7[Dataset Explorer station statistics session patterns geographic analysis network distribution]

  %% =========================
  %% Visualization Layer
  %% =========================
  J0[Visualization Components]
  J1[Plotly Charts bar charts line charts scatter plots gauge charts]
  J2[Folium Maps interactive markers heat maps station clusters distance calculations]
  J3[PyDeck 3D Maps 3D station visualization geographic layers interactive tooltips]

  %% =========================
  %% Flow - Data Ingestion
  %% =========================
  A1 --> B0
  A2 --> B0
  A3 --> B0
  A4 --> B0

  B0 --> B1 --> C1
  B0 --> B2 --> C2
  B0 --> B3 --> C3
  B0 --> B4

  %% =========================
  %% Flow - Data Cleaning
  %% =========================
  C1 --> D0
  C2 --> D0
  C3 --> D0

  D0 --> D1
  D0 --> D2
  D0 --> D3

  %% =========================
  %% Flow - Feature Engineering
  %% =========================
  D1 --> E0
  D2 --> E0
  D3 --> E0

  E0 --> E1 --> E5
  E0 --> E2 --> E5
  E0 --> E3 --> E5
  E0 --> E4 --> E5

  %% =========================
  %% Flow - Model Training
  %% =========================
  E5 --> F0
  F0 --> F1 --> F2
  F2 --> F3 --> F4 --> F5

  %% =========================
  %% Flow - Pricing Engine
  %% =========================
  C2 --> G0
  B4 --> G0
  G0 --> G1 --> G3
  G0 --> G2 --> G3
  G3 --> G4

  %% =========================
  %% Flow - Analytics Engine
  %% =========================
  F5 --> H0
  G4 --> H0
  C2 --> H0

  H0 --> H1
  H0 --> H2
  H0 --> H3
  H0 --> H4

  %% =========================
  %% Flow - Dashboard
  %% =========================
  H0 --> I0

  I0 --> I1
  I0 --> I2
  I0 --> I3
  I0 --> I4
  I0 --> I5
  I0 --> I6
  I0 --> I7

  %% =========================
  %% Flow - Visualization
  %% =========================
  I1 --> J0
  I2 --> J0
  I4 --> J0
  I5 --> J0
  I7 --> J0

  J0 --> J1
  J0 --> J2
  J0 --> J3

  %% =========================
  %% User Interaction Flow
  %% =========================
  J1 --> I0
  J2 --> I0
  J3 --> I0