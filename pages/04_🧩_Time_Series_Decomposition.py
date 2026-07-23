import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
from utils import df , cleaned_df , variable_names ,cleaned_df_1
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(
    page_title ="Time Series Decomposition",
    page_icon="📉",
    layout="wide"
)
st.header("📉 Time Series Decomposition")
st.write("Explore the trend, seasonal, and residual components  of RBI Monetary Indicators ")
variables=st.selectbox(
    "Select Monetary Indicator ",
    list(variable_names.keys()),
    format_func=lambda x:variable_names[x]
)
sep_col="RBIs_Claims_on_Central_Govt"
series3=cleaned_df[variables]
if variables==sep_col :
    st.info(
    "Note: This indicator contains leading missing observations in the original RBI dataset. "
    "Only interpolable values were estimated using linear interpolation, and the decomposition is "
    "performed on the available observations."
    )
    series3=cleaned_df[variables].dropna()
else :
    series3=cleaned_df[variables]

decomposition=seasonal_decompose(series3,model="additive" ,period=52)
fig=decomposition.plot()
fig.set_size_inches(12,12)
fig.set_dpi(120)


fig.suptitle(f"Additive Time Series Decomposition of {variable_names[variables]}")
for ax in fig.axes:
    ax.grid(True,linestyle="--", linewidth=0.5, alpha=0.6)

st.pyplot(fig)















