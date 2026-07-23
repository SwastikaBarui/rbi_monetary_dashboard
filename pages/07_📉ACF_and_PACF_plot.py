import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import df, cleaned_df, variable_names,cleaned_df_1
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.set_page_config(
    page_title="ACF & PACF Plot",
    page_icon="📉",
    layout="wide"
)
st.header("📉 ACF & PACF Plot Of RBI Monetary Indicators ")
st.write(""" 
Explore the Autocorrelation and Partial Autocorrelation structure of the selected RBI Monetary Indicator.
The ACF and PACF plots are calculated dynamically using the verified stationary transformations.
""")

variables = st.selectbox(
    "Select the Monetary Indicator ",
    list(variable_names.keys()),
    format_func=lambda x: variable_names[x]
)
st.markdown(f"### ACF & PACF Analysis Of {variable_names[variables]} ")

# --- 1. PULL CONFIGURATIONS DYNAMICALLY FROM PIPELINE MEMORY ---
use_log = st.session_state.get("use_log", False)
if variables=="Net_foreign_exchange_assets_of_RBI":
    d_value=2
else:
    d_value=1

if d_value == 0:
    diff_label = "No Differencing (d=0)"
elif d_value == 1:
    diff_label = "First-Order Differencing (d=1)"
else:
    diff_label = "Second-Order Differencing (d=2)"

st.info(f"""Integration Order: **{diff_label}**""")

# --- 2. PRE-PROCESS CHOSEN ARRAYS ---
series_1 = cleaned_df_1[variables].dropna()

# Double check log safety logic at runtime
if (series_1 <= 0).any() and use_log:
    st.warning("Data fallback triggered: Selected indicator contains zero or negative values. Bypassing Log.")
    series = series_1
else:
    series = np.log(series_1) if use_log else series_1

# Apply the memory-cached differencing rules
if d_value == 0:
    transformed_var = series
elif d_value == 1:
    transformed_var = series.diff().dropna()
else:
    transformed_var = series.diff().diff().dropna()


lags=st.selectbox(
    "Select the lags ",
    [30,52,60,98]
)
# --- 3. PLOT SIDE-BY-SIDE CANVASES ---
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots(figsize=(8,5))
    plot_acf(transformed_var, lags=lags, ax=ax1)
    ax1.set_title("Autocorrelation Function (ACF)", fontsize=10)
    ax1.set_xlabel("Lags")
    ax1.set_ylabel("Correlation")
    ax1.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig1)
with col2:
    fig2, ax2 = plt.subplots(figsize=(8,5))
    plot_pacf(transformed_var, lags=lags, ax=ax2, method="ywm")
    ax2.set_title("Partial Autocorrelation Function (PACF)", fontsize=10)
    ax2.set_xlabel("Lags")
    ax2.set_ylabel("Partial Correlation")
    ax2.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig2)

st.caption(
    "The blue shaded region represents the 95% confidence interval. Spikes extending beyond this region indicate statistically significant autocorrelations."
)
