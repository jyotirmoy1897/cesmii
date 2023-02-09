"""
Copyright © 2023, Abhishek Hanchate, Satish T.S. Bukkapatnam, and Smart Manufacturing Advanced Research Team, Texas A&M University, All rights reserved.

__author__ = "Abhishek Hanchate"
__copyright__ = "Copyright (C) 2023 Abhishek Hanchate"
__version__ = "1.0"
"""

import streamlit as st
from PIL import Image

# def header(url):
#      st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', \
#                  unsafe_allow_html=True)

def app():

    image = Image.open('simantha_app_logo2.png')
    # st.image(image, use_column_width=True)
    
    col1, col2, col3 = st.beta_columns(3)
    
    with col1:
        st.write(' ')
    
    with col2:
        st.image(image, width=500)
    
    with col3:
        st.write(' ')
    
    st.markdown(
        """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
    st.markdown(""" <style> .font {
        font-size:50px ; font-family: 'Cooper Black'; text-align: centered; color: #500000;} 
        </style> """, unsafe_allow_html=True)
        
    st.markdown('<p class="font">Simantha Simulator Interface</p>', unsafe_allow_html=True)
    
    st.subheader("""
        This app provides visualization and insights for **_Simulated_ Asynchronous Production System Models with finite buffers**!
        """)
    
    st.write('---')









