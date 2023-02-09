# SIMANTHA and Streamlit based Dashboard Script
"""
Copyright © 2023, Abhishek Hanchate, Satish T.S. Bukkapatnam, and Smart Manufacturing Advanced Research Team, Texas A&M University, All rights reserved.

__author__ = "Abhishek Hanchate"
__copyright__ = "Copyright (C) 2023 Abhishek Hanchate"
__version__ = "1.0"
"""

"""
Created on Mon Jun  6 16:16:46 2022
@author: Abhi Hanchate

Example of a condition-based maintenance policy. The CBM threshold determines the health
index level at which a machine requests maintenance. 

Dashboard Script

POTENTIAL INPUTS ON DASHBOARD

Machine:
    degradation_matrix
    cbm_threshold
    pm_distribution
    cm_distribution
    cycle_time

Buffer:
    capacity
    
Maintainer: 
    capacity
    
Simulation:
    simulation_time
    
POTENTIALLY:
    Route: Upstream and Downstream
"""

import pandas as pd
import numpy as np
import plotly.express as px # pip install plotly == 5.2.2            
import plotly.graph_objs as go
from simantha import Source, Machine, Buffer, Sink, Maintainer, System, utils
import streamlit as st
from PIL import Image
import altair as alt


BACKGROUND_COLOR = 'black'
COLOR = 'white'

def set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = False,
        padding_top: int = 1, padding_right: int = 10, padding_left: int = 1, padding_bottom: int = 10,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .reportview-container .css-1lcbmhc .css-1outpf7 {{
                    padding-top: 35px;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )
        

st.set_page_config(
    page_title='My App',
    layout='wide',
    page_icon=':rocket:'
)

set_page_container_style(
        max_width = 1100, max_width_100_percent = True,
        padding_top = 0, padding_right = 10, padding_left = 5, padding_bottom = 10
)



image = Image.open('simantha_app_logo.png')

st.image(image, use_column_width=True)


st.write("""
# Simantha Dashboard
This app provides visualization and insights for **Asynchronous Production System Models with finite buffers.**!
""")
st.write('---')
    

# Sidebar
# Header of Specify Input Parameters

st.sidebar.header('Specify Input Parameters')

def user_input_features():
    Cycle_time = st.sidebar.slider('Cycle Time (in min)', 0, 100, 10)
    Buffer_cap = st.sidebar.slider('Buffer Capacity (in units)', 0, 10, 3)
    Maintainer_cap = st.sidebar.slider('Maintainer Capacity (in units)', 0, 10, 3)
    data = {'Cycle Time (in min)': Cycle_time,
            'Buffer Capacity (in units)': Buffer_cap,
            'Maintainer Capacity (in units)': Maintainer_cap}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

st.header('Which machine would you like to check?')
mc_option = st.selectbox(
     'Select',
     ('Machine1', 'Machine2'))
st.write('You selected:', mc_option)
st.write('---')

degradation_matrix = [
        [0.9, 0.1, 0.,  0.,  0. ],
        [0.,  0.9, 0.1, 0.,  0. ],
        [0.,  0.,  0.9, 0.1, 0. ],
        [0.,  0.,  0.,  0.9, 0.1],
        [0.,  0.,  0.,  0.,  1. ]
]
    
cbm_threshold = 3
pm_distribution = {'geometric': 0.25}
cm_distribution = {'geometric': 0.10}
    
source = Source()
    
M1 = Machine(
        name='M1', 
        cycle_time= int(df['Cycle Time (in min)']),
        degradation_matrix=degradation_matrix,
        cbm_threshold=cbm_threshold,
        pm_distribution=pm_distribution,
        cm_distribution=cm_distribution
)
    
B1 = Buffer(capacity=int(df['Buffer Capacity (in units)']))
    
M2 = Machine(
        name='M2', 
        cycle_time=int(df['Cycle Time (in min)']),
        degradation_matrix=degradation_matrix,
        cbm_threshold=cbm_threshold,
        pm_distribution=pm_distribution,
        cm_distribution=cm_distribution
)
    
sink = Sink()
    
objects = [source, M1, B1, M2, sink]
    
source.define_routing(downstream=[M1])
M1.define_routing(upstream=[source], downstream=[B1])
B1.define_routing(upstream=[M1], downstream=[M2])
M2.define_routing(upstream=[B1], downstream=[sink])
sink.define_routing(upstream=[M2])
    
maintainer = Maintainer(capacity=int(df['Maintainer Capacity (in units)']))
    
system = System(objects=objects, maintainer=maintainer)
    
#random.seed(1)
system.simulate(simulation_time=utils.WEEK)
    
    
  
    
  
M1_prod_df = pd.DataFrame(M1.production_data)
M1_prod_df['Machine'] = "Machine1"                               # Adding Machine ID
M1_prod_df = M1_prod_df.iloc[1: , :]                       # Dropping 0, 0 Row
print(M1_prod_df.head(10))
    
M2_prod_df = pd.DataFrame(M2.production_data)
M2_prod_df['Machine'] = "Machine2"                               # Adding Machine ID
M2_prod_df = M2_prod_df.iloc[1: , :]                       # Dropping 0, 0 Row
print(M2_prod_df.head(10))
    
prod_df = M1_prod_df.append(M2_prod_df)                    # Concatenating DFs
print(prod_df.head(10))
print(prod_df.tail(10))
    
prod_df['prod_rate'] = prod_df['production'] / prod_df['time']  
    
    # prod_df['timediff'] = prod_df.iloc[1: , 1] - prod_df['time']   
    
print(prod_df.head(10))
print(prod_df.tail(10))



M1_health_df = pd.DataFrame(M1.health_data)
M1_health_df['Machine'] = "Machine1"                              # Adding Machine ID
M1_health_df = M1_health_df.iloc[1: , :]                    # Dropping 0, 0 Row
print(M1_health_df.head(10))
    
M2_health_df = pd.DataFrame(M2.health_data)
M2_health_df['Machine'] = "Machine2"                              # Adding Machine ID
M2_health_df = M2_health_df.iloc[1: , :]                    # Dropping 0, 0 Row
print(M2_health_df.head(10))
    
health_df = M1_health_df.append(M2_health_df)                    # Concatenating DFs
print(health_df.head(10))
print(health_df.tail(10))




df = prod_df
df1 = df.groupby(['Machine', 'time'])[['prod_rate']].mean()
df1.reset_index(inplace = True)
print(df1[:5])
    
    
    
    
df2 = df.groupby(['Machine', 'production'])[['time']].mean()
df2.reset_index(inplace = True)
print(df2[:5])




dff = health_df
df3 = dff.groupby(['Machine', 'time'])[['health']].mean()
df3.reset_index(inplace = True)
print(df3[:5])
  


#chart_data = df3[df3['Machine'] == mc_option]
#chart_data2 = chart_data[['time', 'health']]

#st.line_chart(chart_data2, width = 800, height = 600)
# c = alt.Chart(chart_data, title='Machine Health Index across Time').mark_line().encode(
#      x='time', y='health')

# st.altair_chart(c, use_container_width=True)


df_strip = df3[df3["Machine"]==mc_option]
fig_strip = px.line(df_strip, x = 'time', y = 'health', width=800, height=500)
fig_strip.update_xaxes(
        title_text = "<b>Time (Minutes)</b>",
        title_font = dict(size = 22, color='white'),
        title_standoff = 25,
        tickfont=dict(size=18),
        showline=True, linewidth=3, linecolor='white', mirror=True)
fig_strip.update_yaxes(
        title_text = "<b>Health Index</b>",
        title_font = dict(size = 22, color='white'),
        title_standoff = 25,
        tickfont=dict(size=18),
        showline=True, linewidth=4, linecolor='white', mirror=True)
fig_strip.update_traces(line_color='green')
# fig_strip.update_traces(line=dict(color="Maroon", width=2))  
st.plotly_chart(fig_strip, use_container_width=True)
st.write('---')

df_line = df1[df1["Machine"]==mc_option]
fig_line = px.line(df_line, x="time", y="prod_rate", width=800, height=500)
fig_line.update_xaxes(
        title_text = "<b>Time (Minutes)</b>",
        title_font = dict(size = 22, color='white'),
        title_standoff = 25,
        tickfont=dict(size=18),
        showline=True, linewidth=3, linecolor='white', mirror=True)
fig_line.update_yaxes(
        title_text = "<b>Production Rate (Units of Part/Minute) </b>",
        title_font = dict(size = 22, color='white'),
        title_standoff = 25,
        tickfont=dict(size=18),
        showline=True, linewidth=4, linecolor='white', mirror=True)
fig_strip.update_traces(line_color='green')
#fig_line.update_traces(line=dict(color="Maroon", width=2))
st.plotly_chart(fig_line, use_container_width=True)
    
    
    
    
    
