import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import df , cleaned_df,variable_names, cleaned_df_1
from statsmodels.tsa.stattools import adfuller ,kpss


st.set_page_config(
    page_title="Stationarity Test",
    page_icon="✏️",
    layout="wide"
)

st.header("✏️ Stationarity Test")
st.write("Whether the selected Monetary Indicator is Stationary using Augmented Dicky Fuller Test(ADF Test) "
         "& KPSS Test ")
with st.expander("Hypotheses of ADF Test"):
    st.write("""
    **Null Hypothesis (H₀):** The time series contains a unit root (the series is non-stationary).

    **Alternative Hypothesis (H₁):** The time series does not contain a unit root (the series is stationary).
    """)
with st.expander("Hypothesis of KPSS Test") :
    st.write("""
    **Null Hypothesis (H₀):** The series is stationary.

    **Alternative Hypothesis (H₁):** the series is non-stationary.
    """)

variables=st.selectbox(
    "Select Monetary Indicator",
    list(variable_names.keys()),
    format_func=lambda x:variable_names[x]
)
series_1=cleaned_df_1[variables].dropna()
# --- 2. SIMPLE FIX FOR 0 OR NEGATIVE VALUES ---
# Check if any value in the selected series is less than or equal to 0
if (series_1 <= 0).any():
    st.warning(f"Note: {variable_names[variables]} contains 0 or negative values. Bypassing Log Transformation.")
    series = series_1 # Keep data exactly as it is
    st.session_state["use_log"] = False
else:
    series = np.log(series_1)  # Safe to apply Log
    st.session_state["use_log"] = True


sep_col="RBIs_Claims_on_Central_Govt"
if  variables == sep_col :
    st.info("Note: This indicator contains leading missing observations in the original RBI dataset. "
        "Only interpolable values were estimated using linear interpolation, and the decomposition is "
        "performed on the available observations.")

st.markdown(f"### Stationarity Test for ***{variable_names[variables]}***")   
col1 , col2 =st.columns([1,1])
with col1 :
    st.markdown("#### ADF Test Result")
    adf=adfuller(series)
    st.write(f"**ADF Statistic :** {round(adf[0] , 3)}")
    st.write(f"**p-value :** {round(adf[1] , 3)}")
    st.write("**critical values :**")
    for keys,value in adf[4].items():
        st.write(f"{keys}  :     {round(value ,3)}")
with col2:
    st.markdown("#### KPSS Test Result")
    kp_t=kpss(series)
    st.write(f"**KPSS Statistic :** {round(kp_t[0] , 3)}")
    st.write(f"**p-value :** {round(kp_t[1] , 3)}")
    st.write("**critical values :**")
    for keys,value in kp_t[3].items():
        st.write(f"{keys}  :     {round(value ,3)}")


if (adf[1] >0.05 and kp_t[1] <0.05) :
    st.error("**The Time Series is Non-Stationary**")
elif (adf[1]<0.05 and kp_t[1] >0.05) :
    st.error("**The Time Series is Stationary**")
elif (adf[1] <0.05 and kp_t[1]  <0.05) :
    st.error("**The Time Series is Inconclusive**(possible trend stationarity)")
else :
    st.error("**The Time Series is Inconclusive**")

st.markdown("----------------------------------------------------------------------------------------------------------")
st.subheader("***Make The Series Stationary***")
transformation=st.selectbox(
    "Select the Transformation to make the Time Series Stationay",
    ["None","First Differencing","Second Difference"]
)
if transformation=="None" :
    transformed_series=series
elif transformation=="First Differencing":
    transformed_series=series.diff().dropna()
else :
    transformed_series=series.diff().diff().dropna()

adf1=adfuller(transformed_series)
kpss1=kpss(transformed_series)
st.write(f"After doing ***{transformation}*** :")
if adf1[1] < 0.05 and kpss1[1] > 0.05:
    st.success("The Time Series bocames  Stationary")

elif adf1[1] > 0.05 and kpss1[1] < 0.05:
    st.error("The Time Series remains Non-Stationary")

elif adf1[1] < 0.05 and kpss1[1] < 0.05:
    st.warning("The Time Series is Inconclusive (Possible Trend Stationarity)")

else:
    st.warning("The Time Series is Inconclusive")

st.info("""
**Note:** Apply first-order differencing before considering second-order differencing.
If the series becomes stationary after first-order differencing (as indicated by the ADF and KPSS tests),
there is **no need** to apply second-order differencing. Additional differencing may lead to
**over-differencing**, which can remove useful information from the time series.
""")
st.markdown("----------------------------------------------------------------------------------------------------")
fig , ax=plt.subplots(figsize=(12,4))
ax.plot(transformed_series.index,transformed_series.values)
ax.set_title(f"{variable_names[variables]} ({transformation})")
ax.grid(True,linestyle="--",alpha=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("Value")
st.pyplot(fig)