import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Optional: for enhanced visuals
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# Set page configuration and custom CSS for better styling
# ----------------------------
st.set_page_config(page_title="Groundwater Predictor 🌊", layout="wide")
st.markdown(
    """
    <style>
    body {background-color: #f0f2f6; font-family: 'Segoe UI', sans-serif;}
    .main {background-color: #ffffff; padding: 2rem; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1);}
    .stButton>button {background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 16px;}
    .stButton>button:hover {background-color: #388E3C;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Title and description
# ----------------------------
st.title("Groundwater Level Prediction System 🌊")
st.markdown("""
Predict the groundwater levels using historical data, and see a forecast for the next 5 years (2025-2030).  
Select your station code from the dropdown below to view historical trends (including filled data for 2019–2024) and future forecasts 🔮.
""")

# ----------------------------
# Step 1: Data Preprocessing & Feature Engineering
# ----------------------------
@st.cache_data(show_spinner=True)
def load_data():
    data_path = "gwl_daily.csv"  # Adjust path     if needed
    df = pd.read_csv(data_path)
    # Convert measurement date to datetime and extract date features
    df["MSMT_DATE"] = pd.to_datetime(df["MSMT_DATE"])
    df["Year"] = df["MSMT_DATE"].dt.year
    df["Month"] = df["MSMT_DATE"].dt.month
    df["Day"] = df["MSMT_DATE"].dt.day
    df["DayOfYear"] = df["MSMT_DATE"].dt.dayofyear
    df = df.drop(columns=["MSMT_DATE"])
    # Drop quality control columns
    qc_columns = ["WLM_RPE_QC", "WLM_GSE_QC", "RPE_WSE_QC", "GSE_WSE_QC", "WSE_QC"]
    df = df.drop(columns=qc_columns)
    # Normalize numeric columns
    numeric_columns = ["WLM_RPE", "WLM_GSE", "RPE_WSE", "GSE_WSE", "WSE"]
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

df = load_data()

# ----------------------------
# Step 2: Station Selection (Dropdown with Placeholder)
# ----------------------------
station_options = ["Select a Station Code 🚰"] + sorted(df["STATION"].unique().tolist())
selected_station = st.selectbox("Select a Station Code 🚰", station_options)

if selected_station == "Select a Station Code 🚰":
    st.info("Please select a station code to see predictions.")
else:
    st.success(f"You selected station: **{selected_station}** 👍")
    
    # ----------------------------
    # Step 3: Aggregate Data by Year for the Selected Station
    # ----------------------------
    station_data = df[df["STATION"] == selected_station]
    if station_data.empty:
        st.error("No data available for this station. Please select another station.")
    else:
        # Aggregate historical data (yearly average)
        station_yearly = station_data.groupby("Year")["WSE"].mean().reset_index()
        
        # Define desired historical range (2019-2024)
        desired_years = list(range(2019, 2025))
        # Identify missing years in the aggregated historical data
        missing_years = [year for year in desired_years if year not in station_yearly["Year"].values]
        
        # Train a baseline model on available historical data
        X_hist = station_yearly[["Year"]]
        y_hist = station_yearly["WSE"]
        baseline_model = LinearRegression()
        baseline_model.fit(X_hist, y_hist)
        
        # Predict missing years if any exist
        if missing_years:
            pred_missing = pd.DataFrame({"Year": missing_years})
            pred_missing["WSE"] = baseline_model.predict(pred_missing[["Year"]])
            # Append predicted missing years to historical data
            combined_data = pd.concat([station_yearly, pred_missing], ignore_index=True)
        else:
            combined_data = station_yearly.copy()
            
        # Sort the combined data by Year
        combined_data = combined_data.sort_values("Year").reset_index(drop=True)
        
        st.subheader("Historical Data (Yearly Average) 📈")
        st.dataframe(combined_data)
        
        # ----------------------------
        # Step 4: Forecast Future Groundwater Levels from 2025 to 2030
        # ----------------------------
        # Train a new model on the complete historical series
        model_future = LinearRegression()
        model_future.fit(combined_data[["Year"]], combined_data["WSE"])
        
        future_years = pd.DataFrame({"Year": list(range(2025, 2031))})
        future_years["Predicted_WSE"] = model_future.predict(future_years[["Year"]])
        
        st.subheader("Forecast for Next 6 Years (2025-2030) 🔮")
        st.dataframe(future_years)
        
        # ----------------------------
        # Step 5: Visualization
        # ----------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(combined_data["Year"], combined_data["WSE"], marker="o", label="Historical Data (2019-2024 Included)")
        ax.plot(future_years["Year"], future_years["Predicted_WSE"], marker="x", linestyle="--", color="red", label="Forecast (2025-2030)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Normalized WSE")
        ax.set_title(f"Groundwater Level Prediction for Station {selected_station} 🌊")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # ----------------------------
        # Step 6: Flowchart Explanation
        # ----------------------------
        st.subheader("Prediction Process Flowchart 📊")
        flowchart_text = """
        **Flowchart: Groundwater Level Prediction Process**
        
        1. **User Selects Station:** Choose a station code from the dropdown.
        2. **Data Loading & Filtering:** Load the dataset and filter for the selected station.
        3. **Data Aggregation:** Aggregate historical data on a yearly basis.
        4. **Missing Data Filling:** Predict and fill missing years (2019-2024) using a baseline model.
        5. **Model Training:** Train a Linear Regression model on the complete historical series.
        6. **Forecasting:** Predict groundwater levels for 2025-2030.
        7. **Visualization:** Display historical data (with 2019-2024 filled) and forecast in one continuous graph.
        """
        st.markdown(flowchart_text)
