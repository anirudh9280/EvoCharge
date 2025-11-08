import os
import pandas as pd
import streamlit as st
import pydeck as pdk
import numpy as np
import json

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="EvoCharge — San Diego EV Stations", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path configuration - works from any project location
# Get the project root directory (EvoCharge folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "afdc")
TOP50_CSV = os.path.join(DATA_DIR, "afdc_stations_top50.csv")
RAW_CSV = os.path.join(DATA_DIR, "afdc_stations_raw.csv")

# San Diego neighborhood boundaries (simplified GeoJSON-like data)
SD_NEIGHBORHOODS = {
    "Mission Bay": {"lat": 32.7767, "lon": -117.2264, "color": [70, 130, 180, 100]},
    "Point Loma": {"lat": 32.7157, "lon": -117.2447, "color": [60, 179, 113, 100]},
    "Mission Valley": {"lat": 32.7642, "lon": -117.1661, "color": [255, 165, 0, 100]},
    "Downtown": {"lat": 32.7157, "lon": -117.1611, "color": [220, 20, 60, 100]},
    "Balboa Park": {"lat": 32.7341, "lon": -117.1443, "color": [138, 43, 226, 100]},
    "University Heights": {"lat": 32.7489, "lon": -117.1431, "color": [255, 20, 147, 100]},
    "Kensington": {"lat": 32.7631, "lon": -117.1164, "color": [30, 144, 255, 100]},
    "Ocean Beach": {"lat": 32.7467, "lon": -117.2517, "color": [255, 99, 71, 100]},
    "Pacific Beach": {"lat": 32.7964, "lon": -117.2581, "color": [50, 205, 50, 100]},
    "La Jolla": {"lat": 32.8328, "lon": -117.2713, "color": [255, 215, 0, 100]}
}

@st.cache_data
def load_stations(path: str):
    """Load and clean station data from CSV."""
    try:
        df = pd.read_csv(path)
        
        # Ensure required columns exist
        required_cols = ["latitude", "longitude", "station_name", "ev_network", 
                        "ev_dc_fast_num", "ev_level2_evse_num"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # Convert numeric columns safely
        for col in ["ev_dc_fast_num", "ev_level2_evse_num"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        
        # Calculate capacity proxy if not present
        if "capacity_proxy" not in df.columns:
            df["capacity_proxy"] = df["ev_dc_fast_num"] * 1.0 + df["ev_level2_evse_num"] * 0.25
        
        # Clean data - remove rows without coordinates
        df = df.dropna(subset=["latitude", "longitude"]).copy()
        
        # Add some additional useful columns
        df["total_ports"] = df["ev_dc_fast_num"] + df["ev_level2_evse_num"]
        df["has_dc_fast"] = df["ev_dc_fast_num"] > 0
        
        return df
        
    except FileNotFoundError:
        st.error(f"Data file not found: {path}")
        st.info("Make sure you've run the data collection notebook first!")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# -----------------------------
# Load initial data
# -----------------------------
# Try multiple possible locations for the CSV files
possible_paths = [
    (TOP50_CSV, RAW_CSV),  # Primary location: ../data/afdc/
    ("afdc_stations_top50.csv", "afdc_stations_raw.csv"),  # Current directory
    ("data/afdc_stations_top50.csv", "data/afdc_stations_raw.csv"),  # Relative path
]

df = None
for top50_path, raw_path in possible_paths:
    if os.path.exists(top50_path):
        df = load_stations(top50_path)
        TOP50_CSV = top50_path  # Update global paths
        RAW_CSV = raw_path
        st.success(f"✅ Loaded data from: {top50_path}")
        break
    elif os.path.exists(raw_path):
        df = load_stations(raw_path)
        TOP50_CSV = top50_path  # Update global paths  
        RAW_CSV = raw_path
        st.warning(f"⚠️ Top 50 CSV not found, loaded raw data from: {raw_path}")
        break

if df is None:
    st.error("❌ No data files found! Please run the data collection notebook first.")
    st.info("""
    **Expected file locations:**
    - `data/afdc/afdc_stations_top50.csv`
    - `data/afdc/afdc_stations_raw.csv`
    
    **To fix this:**
    1. Make sure you've run the data collection notebook (`data/afdc/data.ipynb`)
    2. Check that the CSV files were created successfully
    3. Run this Streamlit app from the EvoCharge project root directory
    """)
    st.stop()

if df.empty:
    st.error("No valid station data loaded.")
    st.stop()

# -----------------------------
# Header
# -----------------------------
st.title("EvoCharge — San Diego EV Charger Map")
st.markdown("**Real-time EV charging station availability predictor for San Diego County**")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Stations", len(df))
with col2:
    st.metric("Networks", df["ev_network"].nunique())
with col3:
    st.metric("DC Fast Stations", (df["ev_dc_fast_num"] > 0).sum())
with col4:
    st.metric("Avg Capacity Score", f"{df['capacity_proxy'].mean():.1f}")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("🎛️ Map Controls")

# Dataset selection
csv_options = []
if os.path.exists(TOP50_CSV):
    csv_options.append("Top 50 (highest capacity)")
if os.path.exists(RAW_CSV):
    csv_options.append("All stations (raw data)")

if len(csv_options) > 1:
    csv_choice = st.sidebar.selectbox("Dataset", csv_options, index=0)
    if "All stations" in csv_choice:
        df = load_stations(RAW_CSV)
    else:
        df = load_stations(TOP50_CSV)

# Network filter
networks = ["(All Networks)"] + sorted([n for n in df["ev_network"].dropna().unique() if pd.notna(n)])
net_filter = st.sidebar.selectbox("🔌 Network Filter", networks, index=0)

# Capacity filters
st.sidebar.subheader("⚡ Charging Capacity")
min_dc = st.sidebar.slider("Min DC Fast Chargers", 0, int(df["ev_dc_fast_num"].max()), 0)
min_l2 = st.sidebar.slider("Min Level 2 Chargers", 0, int(df["ev_level2_evse_num"].max()), 0)

# Map display options
st.sidebar.subheader("Map Display")
layer_mode = st.sidebar.radio("Visualization Type", ["ChargePoint Style", "Heatmap"], index=0)

# Map style options
show_neighborhoods = st.sidebar.checkbox("Show Neighborhood Labels", value=True)
show_boundaries = st.sidebar.checkbox("Show Area Boundaries", value=True)

if layer_mode == "ChargePoint Style":
    point_size = st.sidebar.slider("Station Marker Size", 100, 300, 180)
    marker_style = st.sidebar.selectbox("Marker Style", 
                                       ["Green Circles (ChargePoint)", "Color by Capacity", "Color by Network"])
else:
    heat_radius = st.sidebar.slider("Heat Radius", 60, 600, 180)

zoom_level = st.sidebar.slider("Zoom Level", 9, 15, 11)

# Future prediction controls (placeholder)
st.sidebar.subheader("Availability Prediction")
eta_minutes = st.sidebar.slider("Arrival Time (minutes from now)", 0, 120, 30, step=15)
st.sidebar.caption("🚧 Prediction model integration coming in Week 4-5")

# Show/hide additional info
show_details = st.sidebar.checkbox("Show Station Details", value=True)

# -----------------------------
# Filter data based on controls
# -----------------------------
mask = (df["ev_dc_fast_num"] >= min_dc) & (df["ev_level2_evse_num"] >= min_l2)

if net_filter != "(All Networks)":
    mask = mask & (df["ev_network"] == net_filter)

df_filtered = df.loc[mask].copy()

if df_filtered.empty:
    st.warning("No stations match your current filters. Try adjusting the criteria.")
    st.stop()

# -----------------------------
# Calculate map center
# -----------------------------
center_lat = float(df_filtered["latitude"].mean())
center_lon = float(df_filtered["longitude"].mean())

# -----------------------------
# Create map layers
# -----------------------------
tooltip = {
    "html": """
    <div style="background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 8px 0; color: #2E7D32;">{station_name}</h4>
        <p style="margin: 2px 0; color: #666;"><b>Network:</b> {ev_network}</p>
        <p style="margin: 2px 0; color: #666;"><b>Address:</b> {street_address}</p>
        <p style="margin: 2px 0; color: #666;"><b>DC Fast:</b> {ev_dc_fast_num} ports | <b>Level 2:</b> {ev_level2_evse_num} ports</p>
        <p style="margin: 2px 0; color: #2E7D32;"><b>Capacity Score:</b> {capacity_proxy:.1f}</p>
    </div>
    """,
    "style": {"backgroundColor": "transparent", "color": "black"}
}

layers = []

# Add neighborhood boundaries if enabled
if show_boundaries:
    # Create simple circular boundaries for neighborhoods
    neighborhood_data = []
    for name, info in SD_NEIGHBORHOODS.items():
        neighborhood_data.append({
            "name": name,
            "coordinates": [info["lon"], info["lat"]],
            "radius": 2000,  # 2km radius
            "color": info["color"]
        })
    
    boundary_layer = pdk.Layer(
        "ScatterplotLayer",
        data=neighborhood_data,
        get_position="coordinates",
        get_radius="radius",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 100],
        get_line_width=2,
        pickable=False,
        stroked=True,
        filled=True,
    )
    layers.append(boundary_layer)

# Add neighborhood labels if enabled
if show_neighborhoods:
    neighborhood_labels = []
    for name, info in SD_NEIGHBORHOODS.items():
        neighborhood_labels.append({
            "name": name,
            "coordinates": [info["lon"], info["lat"]],
            "size": 16
        })
    
    label_layer = pdk.Layer(
        "TextLayer",
        data=neighborhood_labels,
        get_position="coordinates",
        get_text="name",
        get_size="size",
        get_color=[80, 80, 80, 200],
        get_angle=0,
        get_alignment_baseline="'center'",
        pickable=False,
    )
    layers.append(label_layer)

# Main station layer
if layer_mode == "ChargePoint Style":
    if marker_style == "Green Circles (ChargePoint)":
        # ChargePoint-style green circles with white numbers
        df_filtered["station_count"] = 1  # Could be aggregated count in real implementation
        df_filtered["r"] = 46   # ChargePoint green: #2E7D32
        df_filtered["g"] = 125
        df_filtered["b"] = 50
        
        station_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_filtered,
            get_position='[longitude, latitude]',
            get_radius=point_size,
            get_fill_color='[r, g, b, 220]',
            get_line_color=[255, 255, 255, 255],
            get_line_width=3,
            pickable=True,
            auto_highlight=True,
            stroked=True,
            filled=True,
        )
        
        # Add text layer for numbers (simplified - showing total ports)
        df_filtered["port_text"] = df_filtered["total_ports"].astype(str)
        text_layer = pdk.Layer(
            "TextLayer",
            data=df_filtered,
            get_position='[longitude, latitude]',
            get_text="port_text",
            get_size=14,
            get_color=[255, 255, 255, 255],
            get_angle=0,
            get_alignment_baseline="'center'",
            pickable=False,
        )
        layers.extend([station_layer, text_layer])
        
    elif marker_style == "Color by Capacity":
        # Color gradient based on capacity
        values = df_filtered["capacity_proxy"]
        if values.max() > values.min():
            values_norm = (values - values.min()) / (values.max() - values.min())
        else:
            values_norm = pd.Series([0.5] * len(values), index=values.index)
        
        # Green to red gradient (high capacity = green)
        df_filtered["r"] = (255 * (1 - values_norm)).astype(int)
        df_filtered["g"] = (255 * values_norm).astype(int)
        df_filtered["b"] = 30
        
        station_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_filtered,
            get_position='[longitude, latitude]',
            get_radius=point_size,
            get_fill_color='[r, g, b, 200]',
            get_line_color=[255, 255, 255, 150],
            get_line_width=2,
            pickable=True,
            auto_highlight=True,
            stroked=True,
        )
        layers.append(station_layer)
        
    else:  # Color by Network
        # Different colors for each network
        unique_networks = df_filtered["ev_network"].unique()
        network_colors = {}
        colors = [
            [46, 125, 50],    # Green
            [33, 150, 243],   # Blue  
            [255, 152, 0],    # Orange
            [156, 39, 176],   # Purple
            [244, 67, 54],    # Red
            [0, 150, 136],    # Teal
            [121, 85, 72],    # Brown
            [96, 125, 139],   # Blue Grey
        ]
        
        for i, network in enumerate(unique_networks):
            color = colors[i % len(colors)]
            network_colors[network] = color
        
        df_filtered["r"] = df_filtered["ev_network"].map(lambda x: network_colors.get(x, [100, 100, 100])[0])
        df_filtered["g"] = df_filtered["ev_network"].map(lambda x: network_colors.get(x, [100, 100, 100])[1])
        df_filtered["b"] = df_filtered["ev_network"].map(lambda x: network_colors.get(x, [100, 100, 100])[2])
        
        station_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_filtered,
            get_position='[longitude, latitude]',
            get_radius=point_size,
            get_fill_color='[r, g, b, 200]',
            get_line_color=[255, 255, 255, 150],
            get_line_width=2,
            pickable=True,
            auto_highlight=True,
            stroked=True,
        )
        layers.append(station_layer)

else:
    # Heatmap layer
    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=df_filtered,
        get_position='[longitude, latitude]',
        aggregation='"MEAN"',
        get_weight="capacity_proxy",
        radius_pixels=heat_radius,
        intensity=1,
        threshold=0.03,
    )
    layers.append(heat_layer)

# Map view state
view_state = pdk.ViewState(
    latitude=center_lat, 
    longitude=center_lon, 
    zoom=zoom_level, 
    bearing=0, 
    pitch=0
)

# -----------------------------
# Display the map
# -----------------------------
st.subheader(f"📍 San Diego EV Charging Stations ({len(df_filtered)} stations shown)")

# Add style description
if layer_mode == "ChargePoint Style" and marker_style == "Green Circles (ChargePoint)":
    st.caption("🟢 ChargePoint-style interface: Green circles show charging stations with port counts")
elif layer_mode == "ChargePoint Style":
    st.caption("🎨 Enhanced visualization with neighborhood boundaries and labels")
else:
    st.caption("🔥 Heatmap showing station density and capacity")

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style="mapbox://styles/mapbox/light-v11",
)

st.pydeck_chart(deck, use_container_width=True)

# -----------------------------
# Station details and analytics
# -----------------------------
if show_details:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Station Details")
        display_cols = ["station_name", "ev_network", "city", "street_address", 
                       "ev_dc_fast_num", "ev_level2_evse_num", "capacity_proxy"]
        
        # Rename columns for display
        display_df = df_filtered[display_cols].copy()
        display_df.columns = ["Station Name", "Network", "City", "Address", 
                             "DC Fast", "Level 2", "Capacity Score"]
        
        st.dataframe(
            display_df.sort_values("Capacity Score", ascending=False).reset_index(drop=True),
            use_container_width=True
        )
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        # Network distribution
        network_counts = df_filtered["ev_network"].value_counts().head(5)
        st.write("**Top Networks:**")
        for network, count in network_counts.items():
            st.write(f"• {network}: {count} stations")
        
        st.write("**Charging Types:**")
        dc_only = (df_filtered["ev_dc_fast_num"] > 0) & (df_filtered["ev_level2_evse_num"] == 0)
        l2_only = (df_filtered["ev_dc_fast_num"] == 0) & (df_filtered["ev_level2_evse_num"] > 0)
        both = (df_filtered["ev_dc_fast_num"] > 0) & (df_filtered["ev_level2_evse_num"] > 0)
        
        st.write(f"• DC Fast Only: {dc_only.sum()}")
        st.write(f"• Level 2 Only: {l2_only.sum()}")
        st.write(f"• Both Types: {both.sum()}")

# -----------------------------
# Footer and next steps
# -----------------------------
