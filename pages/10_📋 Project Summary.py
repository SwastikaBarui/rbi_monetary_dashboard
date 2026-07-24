import streamlit as st
import numpy as np
import pandas as pd


st.set_page_config(
    page_title="Project Summary",
    page_icon=" 📋 ",
    layout="wide"
)
st.header("📋 Project Summary")
st.divider()

st.subheader("🚧 Challenges Encountered")
st.write(""":one:  The original RBI dataset did not completely follow a regular weekly frequency. 
Financial year-end observations (31 March) were removed and the dataset 
was reindexed to obtain a consistent weekly time series.""")
st.write(""":two: One monetary indicator contained leading 76 missing observations.
Linear interpolation was applied only to missing values surrounded by valid 
observations to avoid introducing artificial information.""")
st.write(""":three: Major policy and economic events such as the 2016 Demonetisation and 
the 2020 COVID-19 pandemic introduced abrupt changes in several monetary indicators. 
These structural breaks reduced the ability of a standard ARIMA model to capture the underlying dynamics.""")
st.write(""":four: Some RBI indicators exhibited sudden spikes , resulting 
in unstable forecasts and extremely large percentage errors , making them unsuitable for meaningful forcast evaluation for this project .

""")

st.divider()
st.subheader("⚠️ Limitations")
st.write(""":one: The analysis uses univariate ARIMA, which models each"
 monetary indicator independently and ignores relationships among variables.""")
st.write(""":two: ARIMA assumes linear relationships and cannot
 adequately capture sudden policy interventions or structural breaks.Forecast accuracy decreases
 when indicators experience abrupt economic shocks.""")

st.divider()
st.subheader("🚀 Future Scope")

st.write("""
:one: Extend the analysis using **VAR** (or VARIMA where appropriate) to model
the relationships among multiple RBI monetary indicators instead of analysing
each indicator independently.
""")

st.write("""
:two: Explore **SARIMA** models for indicators exhibiting seasonal behaviour
to improve forecasting accuracy.
""")



st.divider()
st.subheader("✅ Conclusion")

st.write("""
This project presents a complete workflow for univariate time series analysis of RBI monetary
indicators, covering data preprocessing, trend analysis, stationarity testing, seasonal decomposition,
ARIMA modelling, and forecasting. The analysis also highlights practical challenges such as irregular observations,
 missing values, sudden policy-driven changes, and their impact on forecasting performance. The project provides a foundation for extending the analysis to multivariate time series models in future work.
""")

st.divider()
st.caption("💡 Skills Applied : Python | Numpy | Pandas | Matplotlib | Streamlit | Statsmodels.tsa ")


st.markdown("---")
st.markdown("<h3 style='text-align: center; color: #555555;'>👨‍💻 Developed by Swastika Barui</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888888;'>Data Science & Analytics Portfolio</p>", unsafe_allow_html=True)
