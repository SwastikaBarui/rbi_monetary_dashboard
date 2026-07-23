import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import df, cleaned_df, variable_names,cleaned_df_1
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="Forecasting",
    page_icon="🔮",
    layout="wide"
)

st.header("🔮  Out-of-Sample Forecasting & Performance Validation ")
st.write("""
This section executes the final ARIMA projection based on your custom model selection 
and scores the model's predictive precision against your validation test data partition.
""")
st.info("""
📌 **Important Note on Variable Selection:** 
This project started with 12 variables, but this forecasting page only includes the **8 best variables**. 

I deliberately removed 4 variables (like *RBI Claims on Banks and Commercial Sector*) from the forecasting step for a clear reason: these columns frequently drop to **zero** or have huge, unpredictable sudden jumps due to emergency policy changes. 
When data drops to zero, the percentage error formula (MAPE) breaks down mathematically (it tries to divide by zero and blows up to infinity). Removing them was a conscious decision to keep the application stable, logical, and accurate.
""")

# --- 1. PULL CONFIGURATIONS DYNAMICALLY FROM PIPELINE MEMORY ---
variables = st.selectbox(
    "Select RBI Indicator",
    ["Net_foreign_exchange_assets_of_RBI","Bankers_deposits_with_RBI","Govts_currency_liabilities_to_the_public","Net_non_monetary_liabilities_of_RBI","Reserve_Money_(Liabilities/Components)","Currency_in_circulation_Total","RBIs_Claims_on_Banks_(Including NABARD)","Other_deposits_with_RBI"]
)

if variables=="Net_foreign_exchange_assets_of_RBI":
    d=st.selectbox(
        "Select Order of Differencing ",
        [2]
    )
else :
    d=st.selectbox(
            "Select Order of Differencing ",
            [1,2]
    )
#For lowest AIC and BIC based on lowest MAPE
if variables=="Currency_in_circulation_Total":
    p_1,q_1=3,3
    p_2,q_2=3,3
elif variables=="Other_deposits_with_RBI":
    p_1,q_1=1,3
    p_2,q_2=1,3
elif variables=="Bankers_deposits_with_RBI":
    p_1,q_1=4,5
    p_2,q_2=3,0
elif variables=="Reserve_Money_(Liabilities/Components)": 
    p_1,q_1=2,5
    p_2,q_2=1,5
elif variables=="RBIs_Claims_on_Government(net)":
    p_1,q_1=4,5
    p_2,q_2=5,2
elif variables=="RBIs_Claims_on_Central_Govt": 
    p_1,q_1=6,5
    p_2,q_2=6,5
elif variables== "RBIs_Claims_on_Banks_and_Commercial_sector":
    p_1,q_1=5,4
    p_2,q_2=3,4
elif variables=="RBIs_Claims_on_Banks_(Including NABARD)": 
    p_1,q_1=6,4
    p_2,q_2=5,4
elif variables=="RBIs_claims_on_Commercial_sector_(Excluding NABARD)":
    p_1,q_1=4,6
    p_2,q_2=2,1
elif variables=="Net_foreign_exchange_assets_of_RBI":
    p_1,q_1=1,2
    p_2,q_2=2,1
elif variables=="Govts_currency_liabilities_to_the_public":
    p_1,q_1=6,6
    p_2,q_2=6,6
else : 
    p_1,q_1=2,6
    p_2,q_2=0,1
        

st.subheader(f"Forecast for: {variable_names[variables]} Using ARIMA({p_1}, {d}, {q_1}) according to Lowest AIC")
    
# Process training and testing data splits (80/20)
series = cleaned_df_1[variables].dropna()
train = int(len(series) * 0.8)
train_data = series.iloc[:train]
test_data = series.iloc[train:]
    
# Process data scale dynamically to match stationarity choices
if (series<=0).any() :
    train_processed = (train_data)
else:
     train_processed = np.log(train_data)
     
 # Match trend argument to 'd' to avoid structural convergence crashes
if d == 1:
    operational_trend = 't'
elif d == 2:
    operational_trend = 'n'
else:
    operational_trend = 'c'
# according to lowest AIC
model=ARIMA(train_processed,order=(p_1,d,q_1),trend=operational_trend)
result=model.fit()
forecasting=result.predict(
    start=test_data.index[0],
    end=test_data.index[-1],
    dynamic=False
)
# Process data scale dynamically to match stationarity choices
if (series<=0).any() :
    forecast_series= forecasting
else:
     forecast_series= np.exp(forecasting)






# --- 2. RENDER THE CHRONOLOGICAL CHART ---
fig, ax = plt.subplots(figsize=(12, 6))
        
# Plot the 80% Training Data in Blue
ax.plot(
   train_data.index, train_data.values, 
   linewidth=3, color="#1f77b4", label="Training Data"
        )
        # Plot the 20% Test Data in Light Blue/Teal
ax.plot(
     test_data.index, test_data.values, 
    linewidth=3, color="#17becf", alpha=0.8, label="Actual Test Data"
        )
# Plot the Forecast in Orange (Attaches perfectly to the boundary line)
ax.plot(
    forecast_series.index, forecast_series.values, 
     linewidth=4, color="#ff7f0e", linestyle="--", label="ARIMA Out-of-Sample Forecast"
        )
        
 # Draw a red vertical dashed line showing the exact train-test boundary split
split_date = train_data.index[-1]
ax.axvline(x=split_date, color="red", linestyle=":", linewidth=2, label="Train-Test Split Boundary")
        
 # Styling the graph
ax.set_title(f"Original Levels vs Predictive Forecasting Loop for {variable_names[variables]}")
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Values")
ax.legend()
st.pyplot(fig)

 # --- 3. COMPUTE AND DISPLAY ACCURACY PERFORMANCE METRICS ---
st.subheader("📈 Model Precision & Accuracy Breakdown")

y_true = test_data.values
y_pred = forecast_series.values
        
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
mae = np.mean(np.abs(y_true - y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
# Print metrics side-by-side in custom Streamlit metric elements
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.2f}")
metric_col2.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")
metric_col3.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")

st.divider()


# according to lowest BIC
st.subheader(f"Forecast for: {variable_names[variables]} Using ARIMA({p_2}, {d}, {q_2}) according to Lowest BIC")
model1=ARIMA(train_processed,order=(p_2,d,q_2),trend=operational_trend)
result1=model1.fit()
forecasting1=result1.predict(
    start=test_data.index[0],
    end=test_data.index[-1],
    dynamic=False
)
if (series<=0).any() :
    forecast_series1= forecasting1
else:
     forecast_series1= np.exp(forecasting1)





# --- 2. RENDER THE CHRONOLOGICAL CHART ---
fig, ax = plt.subplots(figsize=(12, 6))
        
# Plot the 80% Training Data in Blue
ax.plot(
   train_data.index, train_data.values, 
   linewidth=3, color="#1f77b4", label="Training Data"
        )
        # Plot the 20% Test Data in Light Blue/Teal
ax.plot(
     test_data.index, test_data.values, 
    linewidth=4, color="#17becf", alpha=0.8, label="Actual Test Data"
        )
# Plot the Forecast in Orange (Attaches perfectly to the boundary line)
ax.plot(
    forecast_series1.index, forecast_series1.values, 
     linewidth=3, color="#ff7f0e", linestyle="--", label="ARIMA Out-of-Sample Forecast"
        )
        
 # Draw a red vertical dashed line showing the exact train-test boundary split
split_date = train_data.index[-1]
ax.axvline(x=split_date, color="red", linestyle=":", linewidth=2, label="Train-Test Split Boundary")
        
 # Styling the graph
ax.set_title(f"Original Levels vs Predictive Forecasting Loop for {variable_names[variables]}")
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Values")
ax.legend()
st.pyplot(fig)

 # --- 3. COMPUTE AND DISPLAY ACCURACY PERFORMANCE METRICS ---
st.subheader("📈 Model Precision & Accuracy Breakdown")

y_true = test_data.values
y_pred = forecast_series1.values
        
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
mae = np.mean(np.abs(y_true - y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
# Print metrics side-by-side in custom Streamlit metric elements
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.2f}")
metric_col2.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")
metric_col3.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
        

        