import streamlit as st
import matplotlib.pyplot as plt
from utils import df,cleaned_df,variable_names,cleaned_df_1

st.set_page_config(
    page_title="Trend Analysis",
    page_icon="📈",
    layout="wide"
)
st.header("📈 Trend Analysis")
st.write("Explore long term movements and trend of RBI monetary indicators")
variables=st.selectbox(
    "Select Monetary Indicators",
    list(variable_names.keys()),
    format_func=lambda x: variable_names[x]
)
series2=cleaned_df[variables]
st.markdown("--------------------------------------------------------------------------------------------------------")
st.subheader("Time Series PLot")
time=st.selectbox(
    "Select Time Scale",
    ["Weekly","Monthly","Yearly","5-yearly"]
)
if time=="Weekly":
    plot_data=series2
elif time=="Monthly":
    plot_data=series2.resample("ME").last()
elif time=="Yearly":
    plot_data=series2.resample("YE").last()
else :
    plot_data=series2.resample("5YE").last()
st.subheader(f"{variable_names[variables]} ({time}) ")
dates=plot_data.index
start_date , end_date =st.select_slider(
    "Select Date Range",
    options=dates,
    value=(dates.min(),dates.max())
)

plot_data=plot_data.loc[start_date : end_date]
 
fig , ax =plt.subplots(figsize=(10,5))
ax.plot(plot_data.index,plot_data.values)
ax.set_title(variable_names[variables])
ax.set_xlabel('Date')
ax.set_ylabel("Value")
ax.grid(linewidth=1,color="red",alpha=0.4)
st.pyplot(fig)

st.markdown("---------------------------------------------------------------------------------------------------------")
st.subheader("📋 Trend Summary")
start_val=plot_data.iloc[0]
end_val=plot_data.iloc[-1]
net_ch=end_val-start_val
growth=(end_val/start_val)
col1 , col2 ,col3 =st.columns([1,1,1])
with col1:
    st.write("***Starting Value :***",start_val)
with col2:
    st.write("***Ending Value :***",end_val)  
with col3:
    st.write("***Net Change:***",net_ch) 

if growth>1 :
    st.write(f"***Overall Growth :*** {variable_names[variables]} became **{growth :.2f}** times larger than the starting value over the time period" )  
elif growth<1:
    st.write(f"***Overall Growth :*** {variable_names[variables]} became **{growth :.2f}** times smaller than the starting value over the time period" )      
else :
    st.write("No change over time period")
