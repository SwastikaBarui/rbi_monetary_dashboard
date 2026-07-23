import streamlit as st
import pandas as pd
import numpy as np
from utils import df, cleaned_df, variable_names, cleaned_df_1
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Hide standard convergence text alerts to keep the screen clean
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Time Series Modelling",
    page_icon="🤖",
    layout="wide"
)

st.header("🤖 Time Series Modelling")
st.write("This section calculates and finds the best ARIMA orders for the selected indicator.")

variables = st.selectbox(
    "Select the Monetary Indicator",
    list(variable_names.keys()),
    format_func=lambda x: variable_names[x]
)

# 1. Prepare clean data at a strict weekly frequency
series = cleaned_df_1[variables].dropna()

if variables=="Currency_in_circulation_Total":
    col1 , col2 =st.columns(2)
    with col1:
        p=st.slider(
                "Select order p",min_value=2,max_value=5,value=3,step=1
                     )
    with col2:
        q=st.slider(
                "Select order q",min_value=2,max_value=5,value=3,step=1
                             )
elif variables=="Other_deposits_with_RBI":
    col1 , col2 =st.columns(2)
    with col1:
        p=st.slider(
                 "Select order p",min_value=0,max_value=4,value=1,step=1
                        )
    with col2:
        q=st.slider(
                "Select order q",min_value=0,max_value=4,value=1,step=1
                  )
elif variables=="Bankers_deposits_with_RBI":
    col1 , col2 =st.columns(2)
    with col1:
        p=st.slider(
                "Select order p",min_value=2,max_value=5,value=3,step=1
                     )
        with col2:
            q=st.slider(
                "Select order q",min_value=0,max_value=6,value=1,step=1
                     )
elif variables=="Reserve_Money_(Liabilities/Components)": 
    col1 , col2 =st.columns(2)
    with col1:
        p=st.slider(
                "Select order p",min_value=0,max_value=3,value=1,step=1
                 )
    with col2:
        q=st.slider(
                "Select order q",min_value=3,max_value=7,value=4,step=1
                             )
elif variables=="RBIs_Claims_on_Government(net)":
    col1 , col2 =st.columns(2)
    with col1:
        p=st.slider(
                "Select order p",min_value=3,max_value=6,value=5,step=1
                     )
    with col2:
        q=st.slider(
                "Select order q",min_value=1,max_value=6,value=5,step=1
                             )
elif variables=="RBIs_Claims_on_Central_Govt": 
    col1 , col2 =st.columns(2)
    with col1:
        p=st.slider(
           "Select order p",min_value=4,max_value=7,value=6,step=1
                   )
    with col2:
         q=st.slider(
            "Select order q",min_value=3,max_value=7,value=6,step=1
                             )
elif variables== "RBIs_Claims_on_Banks_and_Commercial_sector":
    col1 , col2 =st.columns(2)
    with col1:
        p=st.slider(
            "Select order p",min_value=2,max_value=6,value=3,step=1
                    )
    with col2:
        q=st.slider(
            "Select order q",min_value=2,max_value=5,value=3,step=1
                     )
elif variables=="RBIs_Claims_on_Banks_(Including NABARD)": 
    col1 , col2 =st.columns(2)
    with col1:
        p=st.slider(
                        "Select order p",min_value=3,max_value=7,value=4,step=1
                        )
    with col2:
        q=st.slider(
                                "Select order q",min_value=2,max_value=5,value=4,step=1
                         )
elif variables=="RBIs_claims_on_Commercial_sector_(Excluding NABARD)":
    col1 , col2 =st.columns(2)
    with col1:
        p=st.slider(
                    "Select order p",min_value=1,max_value=5,value=2,step=1
                    )
    with col2:
        q=st.slider(
                            "Select order q",min_value=0,max_value=7,value=2,step=1
                     )
elif variables=="Net_foreign_exchange_assets_of_RBI":
    col1 , col2 =st.columns(2)
    with col1:
        p=st.slider(
                "Select order p",min_value=0,max_value=4,value=1,step=1
                )
    with col2:
        q=st.slider(
                        "Select order q",min_value=0,max_value=4,value=1,step=1
                 )
elif variables=="Govts_currency_liabilities_to_the_public":
    col1 , col2 =st.columns(2)
    with col1:
        p=st.slider(
            "Select order p",min_value=4,max_value=8,value=5,step=1
            )
    with col2:
        q=st.slider(
                    "Select order q",min_value=4,max_value=8,value=5,step=1
             )
else : 
    col1 , col2 =st.columns(2)
    with col1:
        p=st.slider(
            "Select order p",min_value=0,max_value=4,value=1,step=1
            )
    with col2:
        q=st.slider(
                    "Select order q",min_value=0,max_value=7,value=1,step=1
             )

d = 2 if variables=="Net_foreign_exchange_assets_of_RBI" else 1
st.subheader(f"ARIMA ({p},{d},{q}) Model :")
train=int(len(series)*0.8)
train_data=series.iloc[:train]
test_data=series.iloc[train:]
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
model=ARIMA(train_processed,order=(p,d,q),trend=operational_trend)
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

y_val=test_data.values
y_pred=forecast_series.values
mape=np.mean(np.abs((y_val-y_pred)/y_val))*100
aic=result.aic
bic=result.bic
col1 , col2 , col3 =st.columns(3)
with col1:
    st.info(f"***AIC value :*** {aic :,.2f}")
with col2:
    st.info(f"***BIC value :*** {bic :,.2f}")
with col3:
    st.info(f"***MAPE :*** {mape :,.2f}%")

# --- ADD THIS TO THE VERY BOTTOM OF YOUR SCRIPT ---

# Clean notebook parameters mapping (Only your 8 chosen variables)
if variables == "Currency_in_circulation_Total":
    p_1, q_1 = 3, 3
    p_2, q_2 = 3, 3
elif variables == "Other_deposits_with_RBI":
    p_1, q_1 = 1, 3
    p_2, q_2 = 1, 3
elif variables == "Bankers_deposits_with_RBI":
    p_1, q_1 = 4, 5
    p_2, q_2 = 3, 0
elif variables == "Reserve_Money_(Liabilities/Components)": 
    p_1, q_1 = 2, 5
    p_2, q_2 = 1, 5
elif variables == "RBIs_claims_on_Commercial_sector_(Excluding NABARD)":
    p_1, q_1 = 4, 6
    p_2, q_2 = 2, 1
elif variables == "Net_foreign_exchange_assets_of_RBI":
    p_1, q_1 = 1, 2
    p_2, q_2 = 2, 1
elif variables == "Govts_currency_liabilities_to_the_public":
    p_1, q_1 = 6, 6
    p_2, q_2 = 6, 6
else: 
    # This automatically handles Net_non_monetary_liabilities_of_RBI
    p_1, q_1 = 2, 6
    p_2, q_2 = 0, 1

st.markdown("---")
st.success("🎯 ** Optimization Analysis Results**")
st.write(f"* Best parameters based on **Lowest AIC & MAPE**: **ARIMA({p_1}, {d}, {q_1})**")
st.write(f"* Best parameters based on **Lowest BIC & MAPE**: **ARIMA({p_2}, {d}, {q_2})**")



# --- ADD THIS TO THE VERY BOTTOM OF YOUR SCRIPT ---

# The 4 variables that we want to explain
removed_variables = [
    "RBIs_Claims_on_Banks_and_Commercial_sector",
    "RBIs_Claims_on_Banks_(Including NABARD)",
    "RBIs_Claims_on_Government(net)",
    "RBIs_Claims_on_Central_Govt"
]

# If the user selects one of these 4, show this simple note
if variables in removed_variables:
    st.markdown("---")
    st.warning(f"""
    📌 **Why this variable is not on the next page:**
    
    You can see that this variable has a very bad error score (**{mape:,.2f}%**). 
    
    This happens because this column represents emergency bank loans or sudden government decisions. These numbers frequently drop to **zero** or jump up violently out of nowhere. 
    
    When data hits zero, the math for the percentage error (MAPE) completely breaks down because it tries to divide by zero. Because this data is too random and breaks the math, **I have removed it from the next page (Forecasting)** to keep the final charts clean and logical.
    """)
