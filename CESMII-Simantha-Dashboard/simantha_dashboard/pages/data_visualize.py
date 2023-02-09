"""
Copyright © 2023, Abhishek Hanchate, Satish T.S. Bukkapatnam, and Smart Manufacturing Advanced Research Team, Texas A&M University, All rights reserved.

__author__ = "Abhishek Hanchate"
__copyright__ = "Copyright (C) 2023 Abhishek Hanchate"
__version__ = "1.0"
"""

import pandas as pd
import numpy as np
import plotly.express as px # pip install plotly == 5.2.2            
import plotly.graph_objs as go
from simantha import Source, Machine, Buffer, Sink, Maintainer, System, utils
import streamlit as st
from PIL import Image
import altair as alt

def app():

    # Sidebar
    # Header of Specify Input Parameters
    st.markdown(""" <style> .fontone {
        font-size:24px ; font-weight:bold; color: #3378EB;} 
        </style> """, unsafe_allow_html=True)
    st.sidebar.markdown('<p class="fontone">Specify Input Parameters</p>', unsafe_allow_html=True)
    st.markdown(""" <style> .fontthree {
        font-size:18px ; font-weight:bold; color: #3378EB;} 
        </style> """, unsafe_allow_html=True)
    
    # Sidebar - Number of Machines 
    st.markdown(""" <style> .fonttwo {
        font-size:18px ; font-weight:bold; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.sidebar.markdown('<p class="fonttwo">Number of Machines</p>', unsafe_allow_html=True)
    with st.sidebar:
        add_radio1 = st.radio(
            "",
            ("One machine", "Two machines")
        )
    
    st.sidebar.write('---') 
    
    
    # Sidebar - Cycle Time of Machines
    if add_radio1 == "One machine":
        st.sidebar.markdown('<p class="fonttwo">Cycle Time (in min)</p>', unsafe_allow_html=True)
        with st.sidebar:
            add_slider1 = st.slider(
                "", 0, 100, 10
            )
            
    elif add_radio1 == "Two machines":
        st.sidebar.markdown('<p class="fonttwo">Cycle Time of Machines 1 and 2 (in min)</p>', unsafe_allow_html=True)
        with st.sidebar:
            add_slider2 = st.slider(
                "   ", 0, 100, 10
            )
            add_slider3 = st.slider(
                "    ", 0, 100, 10
            )
        
    st.sidebar.markdown('<p class="fonttwo">Buffer Capacity (in units)</p>', unsafe_allow_html=True)
    with st.sidebar:
        add_slider4 = st.slider(
            " ", 0, 10, 3
        )
    
    st.sidebar.markdown('<p class="fonttwo">Maintainance Capacity (in units)</p>', unsafe_allow_html=True)
    with st.sidebar:
        add_slider5 = st.slider(
            "  ", 0, 10, 3
        )
        
    if add_radio1 == "One machine":
        data = {'Cycle Time (mins)': [add_slider1],
                'Buffer Capacity (units)': [add_slider4],
                'Maintainance Capacity (units)': [add_slider5]}
        
        df = pd.DataFrame(data)
    
    elif add_radio1 == "Two machines":
        data = {'Cycle Time - Machine 1 (mins)': [add_slider2],
                'Cycle Time - Machine 2 (mins)': [add_slider3],
                'Buffer Capacity (units)': [add_slider4],
                'Maintainance Capacity (units)': [add_slider5]}
        
        df = pd.DataFrame(data) 
        
        
        
    # Sidebar - Simulation Time
    st.sidebar.markdown('<p class="fonttwo">Simulation Time</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        add_radio2 = st.radio(
            "",
            ("One day", "One week", "One month", "One year")
        )
    
    
    # Main Panel
    
    # if one machine
    # Source -> Machine1 -> Sink
    
    # if two machines
    # Source -> Machine1 -> Buffer -> Machine2 -> Sink
    # Source -> Machine2 -> Buffer -> Machine1 -> Sink
    # Source -> Machine1 -> Machine2 -> Sink
    # Source -> Machine2 -> Machine1 -> Sink
    # Source -> Machine1 and Machine2 -> Sink
    
    # if three machines
    
    # if add_radio1 == "One machine":
    # elif add_radio1 == "Two machines":
        
    if add_radio1 == "One machine": 
        
            # Print specified input parameters
        st.markdown('<p class="fontone">Specified Input parameters</p>', unsafe_allow_html=True)
        
        st.markdown(f'<p class="fonttwo">Cycle Time - </p> <p class="fontthree">{add_slider1} minutes</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="fonttwo">Buffer Capacity - </p> <p class="fontthree">{add_slider4} units</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="fonttwo">Maintainance Capacity - </p> <p class="fontthree">{add_slider5} units</p>', unsafe_allow_html=True)
        
        st.write('---')
        
        
        st.markdown('<p class="fontone">Which machine would you like to check?</p>', unsafe_allow_html=True)
        mc_option1 = st.selectbox(
             ' ',
             ('Machine1', ' '))
        st.write('You selected:', mc_option1)
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
                cycle_time= int(add_slider1),
                degradation_matrix=degradation_matrix,
                cbm_threshold=cbm_threshold,
                pm_distribution=pm_distribution,
                cm_distribution=cm_distribution
        )
              
        sink = Sink()
            
        objects = [source, M1, sink]
            
        source.define_routing(downstream=[M1])
        M1.define_routing(upstream=[source], downstream=[sink])
        sink.define_routing(upstream=[M1])
            
        maintainer = Maintainer(capacity=int(add_slider5))
            
        system = System(objects=objects, maintainer=maintainer)
            
        #random.seed(1)
        if add_radio2 == "One day":
            system.simulate(simulation_time=utils.DAY)
            
        elif add_radio2 == "One week":
            system.simulate(simulation_time=utils.WEEK)
        
        elif add_radio2 == "One week":
            system.simulate(simulation_time=utils.MONTH)
            
        else:
            system.simulate(simulation_time=utils.YEAR)
    
        M1_prod_df = pd.DataFrame(M1.production_data)
        M1_prod_df['Machine'] = "Machine1"                               # Adding Machine ID
        M1_prod_df = M1_prod_df.iloc[1: , :]                       # Dropping 0, 0 Row
        print(M1_prod_df.head(10))
             
        M1_prod_df['prod_rate'] = M1_prod_df['production'] / M1_prod_df['time']  
    
        M1_health_df = pd.DataFrame(M1.health_data)
        M1_health_df['Machine'] = "Machine1"                              # Adding Machine ID
        M1_health_df = M1_health_df.iloc[1: , :]                    # Dropping 0, 0 Row
        print(M1_health_df.head(10))
    
        df = M1_prod_df
        df1 = df.groupby(['Machine', 'time'])[['prod_rate']].mean()
        df1.reset_index(inplace = True)
        print(df1[:5])
           
        df2 = df.groupby(['Machine', 'production'])[['time']].mean()
        df2.reset_index(inplace = True)
        print(df2[:5])
    
        dff = M1_health_df
        df3 = dff.groupby(['Machine', 'time'])[['health']].mean()
        df3.reset_index(inplace = True)
        print(df3[:5])
    
        df_strip = df3[df3["Machine"]==mc_option1]
        fig_strip = px.line(df_strip, x = 'time', y = 'health', width=800, height=500)
        fig_strip.update_xaxes(
                title_text = "<b>Time (Minutes)</b>",
                title_font = dict(size = 22, color='#FF9633'),
                title_standoff = 25,
                tickfont=dict(size=18),
                showline=True, linewidth=3, linecolor='white', mirror=True)
        fig_strip.update_yaxes(
                title_text = "<b>Health Index</b>",
                title_font = dict(size = 22, color='#FF9633'),
                title_standoff = 25,
                tickfont=dict(size=18),
                showline=True, linewidth=4, linecolor='white', mirror=True)
        fig_strip.update_traces(line_color='green') 
        st.plotly_chart(fig_strip, use_container_width=True)
        st.write('---')
        
        
        df_line = df1[df1["Machine"]==mc_option1]
        fig_line = px.line(df_line, x="time", y="prod_rate", width=800, height=500)
        fig_line.update_xaxes(
                title_text = "<b>Time (Minutes)</b>",
                title_font = dict(size = 22, color='#FF9633'),
                title_standoff = 25,
                tickfont=dict(size=18),
                showline=True, linewidth=3, linecolor='white', mirror=True)
        fig_line.update_yaxes(
                title_text = "<b>Production Rate (Units of Part/Minute) </b>",
                title_font = dict(size = 22, color='#FF9633'),
                title_standoff = 25,
                tickfont=dict(size=18),
                showline=True, linewidth=4, linecolor='white', mirror=True)
        fig_strip.update_traces(line_color='green')
        st.plotly_chart(fig_line, use_container_width=True)
        
        
    ##################################
    
    elif add_radio1 == "Two machines":
        
            # Print specified input parameters
        st.markdown('<p class="fontone">Specified Input parameters</p>', unsafe_allow_html=True)
        
        st.markdown(f'<p class="fonttwo">Cycle Time M1 - </p> <p class="fontthree">{add_slider2} minutes</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="fonttwo">Cycle Time M2- </p> <p class="fontthree">{add_slider3} minutes</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="fonttwo">Buffer Capacity - </p> <p class="fontthree">{add_slider4} units</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="fonttwo">Maintainance Capacity - </p> <p class="fontthree">{add_slider5} units</p>', unsafe_allow_html=True)
        
        st.write('---')
        
        st.markdown('<p class="fontone">Which machine would you like to check?</p>', unsafe_allow_html=True)
        mc_option2 = st.selectbox(
             '',
             ('Machine1', 'Machine2'))
        st.write('You selected:', mc_option2)
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
                cycle_time= int(add_slider2),
                degradation_matrix=degradation_matrix,
                cbm_threshold=cbm_threshold,
                pm_distribution=pm_distribution,
                cm_distribution=cm_distribution
        )
            
        B1 = Buffer(capacity=int(add_slider4))
            
        M2 = Machine(
                name='M2', 
                cycle_time=int(add_slider3),
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
            
        maintainer = Maintainer(capacity=int(add_slider5))
            
        system = System(objects=objects, maintainer=maintainer)
            
        #random.seed(1)
        if add_radio2 == "One day":
            system.simulate(simulation_time=utils.DAY)
            
        elif add_radio2 == "One week":
            system.simulate(simulation_time=utils.WEEK)
        
        elif add_radio2 == "One week":
            system.simulate(simulation_time=utils.MONTH)
            
        else:
            system.simulate(simulation_time=utils.YEAR)
         
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
        
        df_strip = df3[df3["Machine"]==mc_option2]
        fig_strip = px.line(df_strip, x = 'time', y = 'health', width=800, height=500)
        fig_strip.update_xaxes(
                title_text = "<b>Time (Minutes)</b>",
                title_font = dict(size = 22, color='#FF9633'),
                title_standoff = 25,
                tickfont=dict(size=18),
                showline=True, linewidth=3, linecolor='white', mirror=True)
        fig_strip.update_yaxes(
                title_text = "<b>Health Index</b>",
                title_font = dict(size = 22, color='#FF9633'),
                title_standoff = 25,
                tickfont=dict(size=18),
                showline=True, linewidth=4, linecolor='white', mirror=True)
        fig_strip.update_traces(line_color='green') 
        st.plotly_chart(fig_strip, use_container_width=True)
        st.write('---')
        
        df_line = df1[df1["Machine"]==mc_option2]
        fig_line = px.line(df_line, x="time", y="prod_rate", width=800, height=500)
        fig_line.update_xaxes(
                title_text = "<b>Time (Minutes)</b>",
                title_font = dict(size = 22, color='#FF9633'),
                title_standoff = 25,
                tickfont=dict(size=18),
                showline=True, linewidth=3, linecolor='white', mirror=True)
        fig_line.update_yaxes(
                title_text = "<b>Production Rate (Units of Part/Minute) </b>",
                title_font = dict(size = 22, color='#FF9633'),
                title_standoff = 25,
                tickfont=dict(size=18),
                showline=True, linewidth=4, linecolor='white', mirror=True)
        fig_strip.update_traces(line_color='green')
        st.plotly_chart(fig_line, use_container_width=True) 
