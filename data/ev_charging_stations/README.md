# EV Charging Stations Dataset

## Overview

This dataset contains comprehensive information about electric vehicle charging stations across the United States, including station locations, charger types, network providers, and facility characteristics.

## Dataset Source

- **Source**: Kaggle - EV Charging Stations US
- **Author**: salvatoresaia
- **Files**:
  - `EV_Charging_Stations_Feb82024.xlsx` (65,134 stations)
  - `EV_Charging_Stations_Jan312023.xlsx` (54,238 stations)
- **Geographic Coverage**: United States (all 50 states)
- **Columns**: 13 features per station

## Dataset Schema

| Column             | Type   | Description                                                               |
| ------------------ | ------ | ------------------------------------------------------------------------- |
| Station Name       | string | Name of the charging station facility                                     |
| Street Address     | string | Physical street address of the station                                    |
| City               | string | City where station is located                                             |
| State              | string | State abbreviation (e.g., CA, NY, TX)                                     |
| ZIP                | string | ZIP code of the station location                                          |
| EV Level1 EVSE Num | float  | Number of Level 1 charging ports (120V, slowest)                          |
| EV Level2 EVSE Num | float  | Number of Level 2 charging ports (240V, standard)                         |
| EV DC Fast Count   | float  | Number of DC Fast charging ports (480V+, fastest)                         |
| EV Network         | string | Network provider (e.g., SHELL_RECHARGE, Non-Networked, EV Connect, FLASH) |
| EV Connector Types | string | Available connector types (J1772, CHADEMO, J1772COMBO)                    |
| Access Code        | string | Access type: public or private                                            |
| Access Detail Code | string | Additional access details (often NaN)                                     |
| Facility Type      | string | Type of facility (UTILITY, PARKING_GARAGE, PUBLIC, WORKPLACE, etc.)       |

## Purpose and Usage

### Primary Use: Context and Pricing Estimation

This dataset provides real-world charging station infrastructure data to enhance the energy consumption prediction model with location-based context and cost estimation capabilities.

### Key Applications

1. **Geographic Analysis and Visualization**

   - Map station locations across the United States
   - Create heat maps showing station density by region
   - Identify areas with high/low charging infrastructure availability
   - Analyze distribution of stations by state and city
   - Support location-based filtering in the dashboard

2. **Charger Type Analysis**

   - Examine distribution of Level 1, Level 2, and DC Fast chargers
   - Analyze charger availability by geographic region
   - Understand infrastructure capacity at different locations
   - Compare charger types across network providers
   - Identify trends in charging infrastructure deployment

3. **Network and Facility Characteristics**

   - Analyze market share of different network providers
   - Compare public vs private station accessibility
   - Examine facility type distribution (utilities, parking garages, workplaces)
   - Understand connector type availability and compatibility
   - Identify network-specific patterns and coverage areas

4. **Pricing Model Development**

   - Create pricing estimation rules based on charger type:
     - DC Fast Charging: $0.40-0.60 per kWh (premium, fastest)
     - Level 2 Charging: $0.20-0.30 per kWh (standard, moderate speed)
     - Level 1 Charging: $0.10-0.15 per kWh (economy, slowest)
   - Apply pricing modifiers based on:
     - Network provider (e.g., Shell_Recharge +10%, Non-Networked baseline)
     - Access type (public +15%, private baseline)
     - Geographic location (urban/high-cost areas +20%)
     - Facility type (parking garages +25%, utilities baseline)

5. **Station Selection and Recommendations**
   - Filter stations by user location
   - Display nearby stations with relevant characteristics
   - Show available charger types at each station
   - Provide pricing tier estimates (low/medium/high)
   - Recommend optimal stations based on user preferences

### Integration with Charging Sessions Dataset

This dataset complements the charging sessions data by:

- Providing real station names and locations for user selection
- Enabling charger type selection (Level 2 vs DC Fast)
- Supporting location-based cost estimation
- Adding context to energy consumption predictions
- Facilitating station-to-station comparisons

### Dashboard Implementation

This dataset powers several dashboard features:

1. **Interactive Station Map**

   - Display all stations in selected geographic area
   - Color-code by charger type availability
   - Show station details on hover/click
   - Filter by network, access type, facility type

2. **Station Selector Interface**

   - Search stations by city/state/ZIP
   - List nearby stations with key characteristics
   - Display charger types and availability
   - Show estimated pricing tier

3. **Cost Estimation Engine**

   - Use station characteristics to calculate pricing
   - Compare costs across different station types
   - Show price breakdown by charger type
   - Provide cost-saving recommendations

4. **Comparison Tools**
   - Compare stations in the same area
   - Analyze cost vs speed tradeoffs
   - Show network coverage by region
   - Display facility type options

## Data Statistics

### Geographic Distribution

- **Total Stations**: 65,134 (February 2024 snapshot)
- **States Covered**: All 50 US states
- **Major Markets**: California, New York, Texas, Florida (highest concentration)

### Charger Type Distribution

- **Level 1 Chargers**: Minimal (mostly NaN, rarely deployed)
- **Level 2 Chargers**: Most common (standard charging infrastructure)
- **DC Fast Chargers**: Growing segment (premium fast charging)

### Network Providers

- **Major Networks**: SHELL_RECHARGE, EV Connect, FLASH, ChargePoint
- **Non-Networked**: Significant portion of independent stations
- **Access Types**: Mix of public and private stations

### Facility Types

- **UTILITY**: Utility company-operated stations
- **PARKING_GARAGE**: Public parking facilities
- **PUBLIC**: General public access locations
- **WORKPLACE**: Employer-provided charging
- **Other**: Hotels, retail, restaurants, etc.

## Files in This Directory

- `ev_charging_stations.ipynb` - Jupyter notebook for data loading and exploration
- `README.md` - This documentation file

Note: The actual Excel files are cached by kagglehub and accessed programmatically.

## Next Steps

1. **Exploratory Data Analysis (EDA)**

   - Analyze geographic distribution of stations
   - Examine charger type availability patterns
   - Study network provider market share
   - Investigate facility type distributions
   - Identify data quality issues (NaN values)

2. **Pricing Model Development**

   - Research actual pricing data from major networks
   - Develop pricing estimation formulas
   - Create pricing lookup tables
   - Implement price modifier logic
   - Validate pricing estimates against market rates

3. **Geocoding and Mapping**

   - Convert addresses to latitude/longitude coordinates
   - Prepare data for interactive mapping
   - Create geographic clustering analysis
   - Develop station density calculations
   - Build location-based search functionality

4. **Dashboard Integration**

   - Export processed station data for Streamlit
   - Create station filtering functions
   - Implement location-based queries
   - Build pricing estimation API
   - Develop station comparison features

5. **Data Quality Improvements**
   - Handle missing values (NaN) appropriately
   - Standardize network names and facility types
   - Validate address data
   - Remove duplicates if present
   - Enrich with additional data sources if needed
