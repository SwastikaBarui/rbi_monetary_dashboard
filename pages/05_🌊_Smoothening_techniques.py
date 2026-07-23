import streamlit as st 
import matplotlib.pyplot as plt
from utils import df, cleaned_df , variable_names , cleaned_df_1

st.set_page_config(
    page_title="Smoothening Technique usig SMA",
    page_icon="🌊",
    layout="wide"
)
st.header("🌊 Smotthening Techniques using ***Simple Moving Average(SMA)*** ")
st.write("""
    ***Simple Moving Average (SMA)*** is used to reduce short-term fluctuations in a time series
and reveal its underlying long-term trend.
""")

variables=st.selectbox(
    "Select Monetary Indicator ",
    list(variable_names.keys()),
    format_func=lambda x:variable_names[x]
)

series=cleaned_df[variables].dropna()
window_size=st.slider(
    "Select window size",
    min_value=3,
    max_value=20,
    value=5,
    step=1
)

st.caption("""
**Window Size:** The number of consecutive observations used to calculate each
Simple Moving Average (SMA) value.
""")
sma=series.rolling(window=window_size).mean()

st.subheader(f"SMA of ***{variable_names[variables]}***")
fig , ax =plt.subplots(figsize=(12,5))
ax.plot(
    series.index , series.values,linewidth=7,alpha=0.5,color="red",label="Original Series"
)
ax.plot(sma.index, sma,linewidth=2,alpha=0.6,color="black",label=(f"{window_size}-week SMA"))
ax.set_title(f"Original Series  vs {window_size}-week SMA")
ax.set_xlabel("Week")
ax.grid(True , linestyle="--", alpha=0.5)
ax.legend()
st.pyplot(fig)

st.write("""
**Note:** Simple Moving Average (SMA) is used only for trend visualization and noise reduction.
It **does not** make a time series stationary.
""")