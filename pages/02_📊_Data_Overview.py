import streamlit as st
import pandas as pd 
import numpy as np 
from utils import df, cleaned_df,variable_names ,cleaned_df_1
import matplotlib.pyplot as plt


st.set_page_config(
    page_title=" Data Overview",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Data Overview")
st.markdown("#### Explore the structure and characteristics of RBI Monetary Dataset")
st.markdown("Dataset contains weekly RBI Monetary Indicators ")
st.write("**Rows :** 1007")
st.write("**Columns:** 13")
st.write("**Frequency:** Weekly")
st.markdown("### :one: Data Preview")
st.dataframe(df.head(15))
st.subheader(":two: 💰 Monetary Indicator Overview")

variables=st.selectbox(
    "Select Monetary Indicator",
    list(variable_names.keys()),
    format_func= lambda x: variable_names[x]
)
series=df[variables]
st.write(" **Variable Type :**", series.dtype)
col1 , col2 = st.columns([1,1])
with col1:
    st.subheader("Original Data Statistics ")
    st.write("**Missing Values:**" , series.isna().sum())
    st.write("**Mean:**" , round(series.dropna().mean() ,3))
    st.write("**Median:**" ,(series.dropna().median()))
    st.write("**Standard Deviation:**" , round(series.dropna().std() ,2))
    st.write("**Minimum Value:**" , series.min())
    st.write("**Maximum Value:**" , series.max())
with col2:
    st.subheader("Cleaned Data Statistics ")
    series1=cleaned_df[variables]
    spe_col="RBIs_Claims_on_Central_Govt"
    if variables==spe_col :
        st.info("The first 76 weekly observations are missing in the original RBI dataset. We should interpolate only those which are surrounded by valid data .")
        st.write("**Missing Values:**" , series1.isna().sum())
        st.write("**Mean:**" , round(series1.dropna().mean() ,3))
        st.write("**Median:**" ,(series1.dropna().median()))
        st.write("**Standard Deviation:**" , round(series1.dropna().std() ,2))
        st.write("**Minimum Value:**" , series1.min())
        st.write("**Maximum Value:**" , series1.max())
    else: 
        st.write("**Missing Values:**" , series1.isna().sum())
        st.write("**Mean:**" , round(series1.mean() ,3))
        st.write("**Median:**" ,(series1.median()))
        st.write("**Standard Deviation:**" , round(series1.std() ,2))
        st.write("**Minimum Value:**" , series1.min())
        st.write("**Maximum Value:**" , series1.max())

st.info("""
**Note:** The original RBI dataset contained **1007 observations**, including a few
financial year-end observations recorded on **31st March**, which do not follow the
regular weekly Friday frequency. During preprocessing, these irregular observations
were excluded to create a consistent weekly time series comprising **999 observations**
for subsequent analysis and modelling.
""")

