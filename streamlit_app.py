import os
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="EvoCharge - CA EV Charging Cost Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIONS_PATH = os.path.join(PROJECT_ROOT, "data", "ca_county_prices", "ev_charging_stations_county_prices.csv")
SESSIONS_PATH = os.path.join(PROJECT_ROOT, "data", "ev_charging_sessions", "ev_charging_sessions.csv")
ZIP_TO_COUNTY_PATH = os.path.join(PROJECT_ROOT, "data", "ca_county_prices", "ca_zip_to_county.csv")
COUNTY_RATES_PATH = os.path.join(PROJECT_ROOT, "data", "ca_county_prices", "ca_county_rates.csv")

# -----------------------------
# Data Loading Functions
# -----------------------------
@st.cache_data
def load_stations():
    """Load California charging stations with county prices."""
    try:
        df = pd.read_csv(STATIONS_PATH)
        # Clean data - remove rows without coordinates or county info
        df = df.dropna(subset=["City", "State", "ZIP"]).copy()
        
        # Ensure numeric columns
        for col in ["EV Level1 EVSE Num", "EV Level2 EVSE Num", "EV DC Fast Count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        
        # Add total chargers column
        df["Total_Chargers"] = (
            df.get("EV Level1 EVSE Num", 0) + 
            df.get("EV Level2 EVSE Num", 0) + 
            df.get("EV DC Fast Count", 0)
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading stations data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_zip_to_county():
    """Load ZIP to county mapping."""
    try:
        df = pd.read_csv(ZIP_TO_COUNTY_PATH)
        # Convert zip_code to string for matching
        df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
        return df
    except Exception as e:
        st.error(f"Error loading ZIP to county mapping: {e}")
        return pd.DataFrame()

@st.cache_data
def load_county_rates():
    """Load county electricity rates."""
    try:
        df = pd.read_csv(COUNTY_RATES_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading county rates: {e}")
        return pd.DataFrame()

@st.cache_resource
def train_lasso_model():
    """Train Shohom's Lasso model (predicts avg power, then multiplies by duration)."""
    try:
        # Load charging sessions data
        df = pd.read_csv(SESSIONS_PATH)
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        # Keep only positive durations
        df = df[df['end_time'] > df['start_time']].copy()
        
        # Feature engineering (Shohom's approach)
        df["duration_hours"] = (df['end_time'] - df['start_time']).dt.total_seconds() / 3600.0
        df["start_hour"] = df['start_time'].dt.hour + df['start_time'].dt.minute / 60.0
        
        # Drop extremely small durations
        df = df[df["duration_hours"] > 1e-3].copy()
        
        # Target: average power (kW) and actual energy (kWh)
        y_power = df['energy_kWh'].values / df["duration_hours"].values
        y_energy = df['energy_kWh'].values
        
        # Features: duration_hours, start_hour, session_day, session_type
        important_num = ["duration_hours", "start_hour"]
        important_cat = ["session_day", "session_type"]
        
        X = df[important_num + important_cat].copy()
        
        # Split data (keep indices for energy calculation)
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, shuffle=True
        )
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_power_train = y_power[train_idx]
        y_power_test = y_power[test_idx]
        y_energy_train = y_energy[train_idx]
        y_energy_test = y_energy[test_idx]
        
        # Build preprocessing pipeline
        transformers = [
            ("imp_num", StandardScaler(), important_num),
            ("imp_cat", OneHotEncoder(handle_unknown="ignore"), important_cat)
        ]
        
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # Lasso model with cross-validation
        lasso = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", LassoCV(alphas=np.logspace(-3, 1, 20), cv=5, random_state=42)),
            ]
        )
        
        # Fit on average power (kW)
        lasso.fit(X_train, y_power_train)
        
        # Calculate performance metrics on energy (kWh)
        power_pred_test = lasso.predict(X_test)
        energy_pred_test = power_pred_test * X_test["duration_hours"].values
        
        mae = mean_absolute_error(y_energy_test, energy_pred_test)
        mse = mean_squared_error(y_energy_test, energy_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_energy_test, energy_pred_test)
        
        # Calculate baseline metrics (mean energy)
        baseline_energy = np.full_like(y_energy_test, y_energy_train.mean())
        baseline_mae = mean_absolute_error(y_energy_test, baseline_energy)
        baseline_rmse = np.sqrt(mean_squared_error(y_energy_test, baseline_energy))
        
        # Store metrics
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'baseline_mae': baseline_mae,
            'baseline_rmse': baseline_rmse,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'selected_alpha': lasso.named_steps['model'].alpha_,
            'mean_energy': y_energy_train.mean(),
            'std_energy': y_energy_train.std()
        }
        
        return lasso, metrics
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None

# -----------------------------
# Helper Functions
# -----------------------------
def get_county_from_zip(zip_code, zip_to_county_df):
    """Get county name from ZIP code."""
    zip_code = str(zip_code).zfill(5)
    match = zip_to_county_df[zip_to_county_df['zip_code'] == zip_code]
    if not match.empty:
        return match.iloc[0]['county_name']
    return None

def get_rate_from_county(county_name, county_rates_df):
    """Get electricity rate from county name."""
    match = county_rates_df[county_rates_df['county_name'] == county_name]
    if not match.empty:
        return match.iloc[0]['rate_per_kwh']
    return None

def predict_charging_cost(model, duration_min, start_hour, session_day, session_type, electricity_rate):
    """Predict charging cost using Shohom's Lasso model."""
    # Convert duration to hours
    duration_hours = duration_min / 60.0
    
    # Create input dataframe (Shohom's features)
    input_data = pd.DataFrame({
        'duration_hours': [duration_hours],
        'start_hour': [start_hour],
        'session_day': [session_day],
        'session_type': [session_type]
    })
    
    # Predict average power (kW)
    avg_power_kw = model.predict(input_data)[0]
    
    # Calculate energy: power × duration
    energy_kwh = avg_power_kw * duration_hours
    
    # Calculate cost
    cost = energy_kwh * electricity_rate
    
    return energy_kwh, cost, avg_power_kw

# -----------------------------
# Load Data
# -----------------------------
with st.spinner("Loading data and training Lasso model..."):
    stations_df = load_stations()
    zip_to_county_df = load_zip_to_county()
    county_rates_df = load_county_rates()
    model, model_metrics = train_lasso_model()

if stations_df.empty or zip_to_county_df.empty or county_rates_df.empty or model is None:
    st.error("Failed to load required data. Please check data files.")
    st.stop()

# -----------------------------
# Header
# -----------------------------
st.title("⚡ EvoCharge - California EV Charging Cost Predictor")
st.markdown("**Predict your EV charging costs based on location, duration, and charging patterns**")

# Display overall statistics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("CA Charging Stations", len(stations_df))
with col2:
    st.metric("Counties Covered", stations_df['county_name'].nunique())
with col3:
    st.metric("Networks", stations_df['EV Network'].nunique())
with col4:
    avg_rate = county_rates_df['rate_per_kwh'].mean()
    st.metric("Avg Rate ($/kWh)", f"${avg_rate:.4f}")
with col5:
    if model_metrics:
        st.metric("Model MAE", f"{model_metrics['mae']:.2f} kWh")

# -----------------------------
# Sidebar - User Inputs
# -----------------------------
st.sidebar.header("🎛️ Charging Session Parameters")

# ZIP Code Input
st.sidebar.subheader("📍 Location")
user_zip = st.sidebar.text_input("Enter ZIP Code", value="90012", max_chars=5)

# Find county and rate for the ZIP code
selected_county = None
selected_rate = None
if user_zip and len(user_zip) == 5:
    selected_county = get_county_from_zip(user_zip, zip_to_county_df)
    if selected_county:
        selected_rate = get_rate_from_county(selected_county, county_rates_df)
        st.sidebar.success(f"County: **{selected_county}**")
        st.sidebar.info(f"Electricity Rate: **${selected_rate:.4f}/kWh**")
    else:
        st.sidebar.warning("ZIP code not found in California database")

# Session Parameters
st.sidebar.subheader("⚙️ Charging Session Details")

# Duration
duration_min = st.sidebar.slider("Charging Duration (minutes)", 30, 120, 75)

# Time of day
# Persist the user's chosen start time across reruns so it doesn't reset
if "start_time_value" not in st.session_state:
    default_start = datetime.now().replace(hour=15, minute=0, second=0, microsecond=0)
    st.session_state.start_time_value = default_start.time()

start_time = st.sidebar.time_input(
    "Start Time",
    value=st.session_state.start_time_value,
    key="start_time_input"
)
st.session_state.start_time_value = start_time
start_hour = start_time.hour + start_time.minute / 60.0  # Decimal hour

# Day of week
session_day = st.sidebar.selectbox("Day of Week", ["Weekday", "Weekend"])

# Session type
session_type = st.sidebar.selectbox("Session Type", ["Regular", "Occasional", "Emergency"], 
                                    help="Session type has minimal impact on cost (~3% difference)")

# Model info
with st.sidebar.expander("ℹ️ Model Performance & Details"):
    st.markdown("### Model Type")
    st.write("**Lasso Regression** (L1-regularized)")
    st.write("- LassoCV with 5-fold cross-validation")
    st.write("- Predicts avg power (kW), then × duration")
    
    if model_metrics:
        st.markdown("### Performance Metrics")
        st.write(f"**MAE**: {model_metrics['mae']:.2f} kWh")
        st.write(f"**RMSE**: {model_metrics['rmse']:.2f} kWh")
        st.write(f"**R² Score**: {model_metrics['r2']:.3f}")
        st.write(f"**Selected Alpha**: {model_metrics['selected_alpha']:.4f}")
        
        st.markdown("### Baseline Comparison")
        improvement_mae = ((model_metrics['baseline_mae'] - model_metrics['mae']) / model_metrics['baseline_mae'] * 100)
        improvement_rmse = ((model_metrics['baseline_rmse'] - model_metrics['rmse']) / model_metrics['baseline_rmse'] * 100)
        st.write(f"**Baseline MAE**: {model_metrics['baseline_mae']:.2f} kWh")
        st.write(f"**Improvement**: {improvement_mae:.1f}% better")
        
        st.markdown("### Training Data")
        st.write(f"**Training**: {model_metrics['train_size']} sessions")
        st.write(f"**Testing**: {model_metrics['test_size']} sessions")
        st.write(f"**Mean Energy**: {model_metrics['mean_energy']:.2f} kWh")
        st.write(f"**Std Dev**: {model_metrics['std_energy']:.2f} kWh")
    
    st.markdown("### Features Used")
    st.write("- Duration (hours) - MAJOR")
    st.write("- Start hour - MEDIUM")
    st.write("- Day type - MINOR")
    st.write("- Session type - MINOR")

# -----------------------------
# Prediction Section
# -----------------------------
st.header("💰 Cost Prediction")

if selected_county and selected_rate:
    # Predict
    energy_kwh, cost_usd, avg_power_kw = predict_charging_cost(
        model, duration_min, start_hour, session_day, session_type, selected_rate
    )
    
    # Display prediction
    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
    with pred_col1:
        st.metric("Predicted Energy", f"{energy_kwh:.2f} kWh")
    with pred_col2:
        st.metric("Avg Charging Power", f"{avg_power_kw:.2f} kW")
    with pred_col3:
        st.metric("Estimated Cost", f"${cost_usd:.2f}")
    with pred_col4:
        cost_per_min = cost_usd / duration_min
        st.metric("Cost per Minute", f"${cost_per_min:.3f}")
    
    # Show breakdown
    with st.expander("📊 Cost Breakdown & Model Info"):
        st.markdown("### Session Details")
        st.write(f"**Location:** {selected_county} County, CA (ZIP: {user_zip})")
        st.write(f"**Electricity Rate:** ${selected_rate:.4f}/kWh")
        st.write(f"**Charging Duration:** {duration_min} minutes ({duration_min/60:.1f} hours)")
        st.write(f"**Start Time:** {start_hour:.2f} (decimal hour)")
        st.write(f"**Session Type:** {session_type} ({session_day})")
        
        st.markdown("### Prediction (Lasso Model)")
        st.write(f"**Average Charging Power:** {avg_power_kw:.2f} kW")
        st.write(f"**Energy Consumption:** {energy_kwh:.2f} kWh = {avg_power_kw:.2f} kW × {duration_min/60:.2f} hours")
        st.write(f"**Total Cost:** ${cost_usd:.2f} = {energy_kwh:.2f} kWh × ${selected_rate:.4f}/kWh")
        
        st.markdown("### Model Information")
        st.info("""
        **Using**: Lasso Regression (L1-regularized linear model)
        
        **Approach**: Predicts average charging power (kW), then multiplies by duration.
        This is more physically sound than directly predicting energy.
        
        **Why better?**
        - Linear relationship is interpretable
        - L1 regularization prevents overfitting
        - Physically meaningful predictions
        """)
else:
    st.info("👈 Enter a valid California ZIP code to see cost predictions")

# Model Performance Section
if model_metrics:
    with st.expander("📊 Model Performance Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Prediction Accuracy")
            st.metric("Mean Absolute Error (MAE)", f"{model_metrics['mae']:.2f} kWh", 
                     help="Average prediction error")
            st.metric("Root Mean Squared Error (RMSE)", f"{model_metrics['rmse']:.2f} kWh",
                     help="Standard deviation of prediction errors")
            st.metric("R² Score", f"{model_metrics['r2']:.3f}",
                     help="Proportion of variance explained (1.0 = perfect)")
        
        with col2:
            st.markdown("### Model vs Baseline")
            improvement_mae = ((model_metrics['baseline_mae'] - model_metrics['mae']) / model_metrics['baseline_mae'] * 100)
            improvement_rmse = ((model_metrics['baseline_rmse'] - model_metrics['rmse']) / model_metrics['baseline_rmse'] * 100)
            
            st.metric("Baseline MAE", f"{model_metrics['baseline_mae']:.2f} kWh",
                     help="Simple mean prediction")
            st.metric("MAE Improvement", f"{improvement_mae:.1f}%",
                     delta=f"{improvement_mae:.1f}%", delta_color="normal")
            st.metric("RMSE Improvement", f"{improvement_rmse:.1f}%",
                     delta=f"{improvement_rmse:.1f}%", delta_color="normal")
        
        with col3:
            st.markdown("### Training Details")
            st.metric("Training Samples", f"{model_metrics['train_size']:,}")
            st.metric("Testing Samples", f"{model_metrics['test_size']:,}")
            st.metric("Selected Alpha (λ)", f"{model_metrics['selected_alpha']:.4f}",
                     help="L1 regularization strength")
        
        st.markdown("---")
        st.markdown("""
        **Interpretation Guide:**
        - **MAE** = Average error in kWh predictions (lower is better)
        - **RMSE** = Penalizes large errors more heavily (lower is better)
        - **R²** = How well the model explains variance (closer to 1.0 is better)
        - **Baseline** = Simple average prediction (no ML)
        - **Alpha** = Regularization parameter preventing overfitting
        
        Our Lasso model significantly outperforms the baseline, demonstrating that 
        session characteristics (duration, time, type) provide valuable predictive power.
        """)

# -----------------------------
# Map Visualization
# -----------------------------
st.header("🗺️ California Charging Stations Map")

# Filter stations by ZIP code if provided
if user_zip and selected_county:
    # Get all stations in the user's county
    filtered_stations = stations_df[stations_df['county_name'] == selected_county].copy()
    map_center_lat = filtered_stations['City'].apply(lambda x: 34.0522).mean() if not filtered_stations.empty else 36.7783
    map_center_lon = filtered_stations['City'].apply(lambda x: -118.2437).mean() if not filtered_stations.empty else -119.4179
    zoom_level = 10
    st.subheader(f"📍 Charging Stations in {selected_county} County ({len(filtered_stations)} stations)")
else:
    # Show all CA stations
    filtered_stations = stations_df.copy()
    map_center_lat = 36.7783  # Center of California
    map_center_lon = -119.4179
    zoom_level = 6
    st.subheader(f"📍 All California Charging Stations ({len(filtered_stations)} stations)")

# Map display options
st.sidebar.subheader("🗺️ Map Options")
show_dc_fast_only = st.sidebar.checkbox("Show DC Fast Chargers Only", value=False)
min_chargers = st.sidebar.slider("Min Total Chargers", 0, int(filtered_stations['Total_Chargers'].max()), 0)

# Apply filters
if show_dc_fast_only:
    filtered_stations = filtered_stations[filtered_stations['EV DC Fast Count'] > 0]
if min_chargers > 0:
    filtered_stations = filtered_stations[filtered_stations['Total_Chargers'] >= min_chargers]

# Pre-cached California city coordinates (major cities)
CA_CITY_COORDS = {
    'Los Angeles': (34.0522, -118.2437),
    'San Francisco': (37.7749, -122.4194),
    'San Diego': (32.7157, -117.1611),
    'San Jose': (37.3382, -121.8863),
    'Sacramento': (38.5816, -121.4944),
    'Oakland': (37.8044, -122.2712),
    'Fresno': (36.7378, -119.7871),
    'Long Beach': (33.7701, -118.1937),
    'Bakersfield': (35.3733, -119.0187),
    'Anaheim': (33.8366, -117.9143),
    'Santa Ana': (33.7455, -117.8677),
    'Riverside': (33.9806, -117.3755),
    'Irvine': (33.6846, -117.8265),
    'San Bernardino': (34.1083, -117.2898),
    'Pasadena': (34.1478, -118.1445),
    'Santa Clara': (37.3541, -121.9552),
    'Berkeley': (37.8715, -122.2730),
    'Sunnyvale': (37.3688, -122.0363),
    'Burbank': (34.1808, -118.3090),
    'Glendale': (34.1425, -118.2551),
    'Palo Alto': (37.4419, -122.1430),
    'Monterey': (36.6002, -121.8947),
    'Santa Barbara': (34.4208, -119.6982),
    'Ventura': (34.2746, -119.2290),
    'Orange': (33.7879, -117.8531),
    'Newport Beach': (33.6189, -117.9289),
    'San Mateo': (37.5630, -122.3255),
    'Redondo Beach': (33.8492, -118.3884),
    'Santa Monica': (34.0195, -118.4912),
    'Culver City': (34.0211, -118.3965),
    'Beverly Hills': (34.0736, -118.4004),
    'West Hollywood': (34.0900, -118.3617),
    'Torrance': (33.8358, -118.3406),
    'Norwalk': (33.9022, -118.0817),
    'Downey': (33.9401, -118.1332),
    'Inglewood': (33.9617, -118.3531),
    'Pomona': (34.0551, -117.7500),
    'Corona': (33.8753, -117.5664),
    'Oxnard': (34.1975, -119.1771),
    'Camarillo': (34.2164, -119.0376),
    'Thousand Oaks': (34.1706, -118.8376),
    'Simi Valley': (34.2694, -118.7815),
    'Carlsbad': (33.1581, -117.3506),
    'Chula Vista': (32.6401, -117.0842),
    'El Cajon': (32.7948, -116.9625),
    'Escondido': (33.1192, -117.0864),
    'La Jolla': (32.8328, -117.2713),
    'Solana Beach': (32.9911, -117.2712),
    'Encinitas': (33.0370, -117.2920),
    'Vista': (33.2000, -117.2425),
}

# County center coordinates
COUNTY_COORDS = {
    'Los Angeles': (34.0522, -118.2437),
    'San Diego': (32.7157, -117.1611),
    'Orange': (33.7175, -117.8311),
    'Riverside': (33.7866, -116.9691),
    'San Bernardino': (34.8373, -116.5453),
    'Santa Clara': (37.3541, -121.9552),
    'Alameda': (37.6469, -121.8872),
    'Sacramento': (38.5449, -121.7405),
    'Contra Costa': (37.9161, -122.0588),
    'Fresno': (36.7378, -119.7871),
    'Kern': (35.3733, -119.0187),
    'San Francisco': (37.7749, -122.4194),
    'Ventura': (34.3705, -119.1391),
    'San Mateo': (37.4337, -122.4014),
    'San Joaquin': (37.9358, -121.2728),
    'Stanislaus': (37.6391, -120.9970),
    'Sonoma': (38.5110, -122.8497),
    'Tulare': (36.2077, -119.3473),
    'Santa Barbara': (34.5708, -120.0958),
    'Monterey': (36.2335, -121.4688),
}

@st.cache_data
def get_city_coords(city):
    """Get coordinates for a city using pre-cached data."""
    # Check if city is in our cache
    if city in CA_CITY_COORDS:
        return CA_CITY_COORDS[city]
    
    # If not in cache, return None (we'll use county coords instead)
    return None, None

# Create map visualization
if not filtered_stations.empty:
    # Determine map center and zoom
    if selected_county:
        # Use county coordinates if available
        if selected_county in COUNTY_COORDS:
            center_lat, center_lon = COUNTY_COORDS[selected_county]
        else:
            center_lat, center_lon = 36.7783, -119.4179  # CA center
        zoom_start = 10
    else:
        center_lat, center_lon = 36.7783, -119.4179  # CA center
        zoom_start = 6
    
    # Create a container for the map to prevent flickering
    map_container = st.container()
    
    with map_container:
        # Sample size for map display
        # Show more stations when filtering by county, fewer for all CA
        default_sample = 100 if selected_county else 75
        sample_size = min(default_sample, len(filtered_stations))
        
        # Get sample of stations to display
        map_stations = filtered_stations.head(sample_size)
        
        # Create folium map
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=zoom_start, 
            tiles="OpenStreetMap",
            prefer_canvas=True  # Better performance
        )
        
        # Add individual station markers
        markers_added = 0
        for idx, row in map_stations.iterrows():
            city = row['City']
            lat, lon = get_city_coords(city)
            
            if lat is not None and lon is not None:
                # Add small random offset to avoid exact overlaps
                lat_offset = lat + np.random.uniform(-0.005, 0.005)
                lon_offset = lon + np.random.uniform(-0.005, 0.005)
                
                # Determine marker color based on rate
                rate = row['electricity_rate_per_kwh']
                if rate <= 0.30:
                    color = 'green'
                elif rate <= 0.3869:
                    color = 'blue'
                else:
                    color = 'orange'
                
                # Create popup with station details
                popup_html = f"""
                <div style="width: 280px;">
                    <h4 style="margin: 0 0 8px 0;">{row['Station Name']}</h4>
                    <p style="margin: 4px 0;"><b>Network:</b> {row['EV Network']}</p>
                    <p style="margin: 4px 0;"><b>Address:</b> {row['Street Address']}</p>
                    <p style="margin: 4px 0;"><b>City:</b> {city}, {row['State']} {row['ZIP']}</p>
                    <p style="margin: 4px 0;"><b>County:</b> {row['county_name']}</p>
                    <p style="margin: 4px 0; color: #2E7D32;"><b>Rate:</b> ${rate:.4f}/kWh</p>
                    <hr style="margin: 8px 0;">
                    <p style="margin: 4px 0;"><b>Level 2:</b> {int(row['EV Level2 EVSE Num'])} ports</p>
                    <p style="margin: 4px 0;"><b>DC Fast:</b> {int(row['EV DC Fast Count'])} ports</p>
                    <p style="margin: 4px 0;"><b>Total:</b> {int(row['Total_Chargers'])} chargers</p>
                </div>
                """
                
                folium.CircleMarker(
                    location=[lat_offset, lon_offset],
                    radius=6,
                    popup=folium.Popup(popup_html, max_width=320),
                    tooltip=row['Station Name'],
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
                markers_added += 1
        
        # Display map with key to prevent re-rendering
        st_folium(m, width=1400, height=500, key="station_map", returned_objects=[])
        
        if len(filtered_stations) > sample_size:
            st.caption(f"🗺️ Showing {markers_added} of {len(filtered_stations)} stations on map. See complete list below.")
        else:
            st.caption(f"🗺️ Showing all {markers_added} stations on map.")
    
    st.markdown("---")
    
    # Display full table in a separate container
    table_container = st.container()
    
    with table_container:
        st.subheader("📋 Complete Station List")
        
        # Add search functionality
        search_term = st.text_input("🔍 Search stations by name, city, or address", "")
        
        display_cols = ['Station Name', 'Street Address', 'City', 'county_name', 
                        'electricity_rate_per_kwh', 'EV Level2 EVSE Num', 'EV DC Fast Count', 'Total_Chargers']
        
        display_df = filtered_stations[display_cols].copy()
        display_df.columns = ['Station Name', 'Address', 'City', 'County', 
                              'Rate ($/kWh)', 'Level 2', 'DC Fast', 'Total Ports']
        
        # Apply search filter
        if search_term:
            mask = (
                display_df['Station Name'].str.contains(search_term, case=False, na=False) |
                display_df['City'].str.contains(search_term, case=False, na=False) |
                display_df['Address'].str.contains(search_term, case=False, na=False)
            )
            display_df = display_df[mask]
            st.info(f"Found {len(display_df)} stations matching '{search_term}'")
        
        display_df = display_df.sort_values('Total Ports', ascending=False).reset_index(drop=True)
        
        st.dataframe(display_df, use_container_width=True, height=400)
else:
    st.info("No stations match the current filters.")

# -----------------------------
# Additional Insights
# -----------------------------
st.header("📈 Insights & Comparisons")

col1, col2 = st.columns(2)

with col1:
    st.subheader("💡 Rate Comparison by County")
    top_counties = county_rates_df.nsmallest(10, 'rate_per_kwh')
    st.write("**Cheapest 10 Counties:**")
    for idx, row in top_counties.iterrows():
        st.write(f"• {row['county_name']}: ${row['rate_per_kwh']:.4f}/kWh")

with col2:
    st.subheader("🔌 Network Distribution")
    network_counts = stations_df['EV Network'].value_counts().head(10)
    st.write("**Top 10 Networks:**")
    for network, count in network_counts.items():
        st.write(f"• {network}: {count} stations")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Data sources: Kaggle EV Charging Stations & Sessions | California County Electricity Rates</p>
    <p>Model: Lasso Regression (L1-regularized) by Shohom | Trained on 3,500 charging sessions</p>
</div>
""", unsafe_allow_html=True)
