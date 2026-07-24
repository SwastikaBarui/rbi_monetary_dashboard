import streamlit as st

st.set_page_config(
    page_title="RBI Dashboard ",
    page_icon="🏦",
    layout="wide",
)
st.title("🏦 RBI Monetary Time Series Analysis Dashboard")
st.subheader("Interactive Time Series Dashboard for RBI Monetary Indicators  ")
col1 , col2 , col3 = st.columns([1,2,1])
with col2:
    st.image("pic.png",width=250)

st.divider()
st.header("📖 About the Dashboard")
st.markdown(" This dashboard provides an interactive analysis of RBI monetary indicators using time series techniques. It enables users to explore trends, examine stationarity, perform decomposition, apply smoothing methods, and build time series models for forecasting.")
st.divider()
col1 , col2 =st.columns([1,1])
with col1:
    st.subheader("📂 Dataset Information")
    st.write("**• Source:** Reserve Bank Of India(RBI)")
    st.write("**• Frequency:** Weekly")
    st.write("**• Observations:** 1008")
    st.write("**• Variables:** 13 Monetray Indicators")
    st.write("**• File Format:** CSV")
    st.write("**• Time Period:** 06 Jul 2001 - 21 Aug 2020")
with col2:
    st.subheader("🚀 Dashboard Features")
    st.write("✔ Data Overview")
    st.write("✔ Trend Analysis")  
    st.write("✔ Time Series Decomposition")
    st.write("✔ Smoothing Techniques")
    st.write("✔ Stationarity Test")  
    st.write("✔ ACF & PACF Analysis")
    st.write("✔ Time Series Modelling")
    st.write("✔ Forecasting")
    st.write("✔ Project Summary")

st.markdown("---")
st.markdown("<h3 style='text-align: center; color: #555555;'>👨‍💻 Developed by Swastika Barui</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888888;'>Data Science & Analytics Portfolio</p>", unsafe_allow_html=True)


