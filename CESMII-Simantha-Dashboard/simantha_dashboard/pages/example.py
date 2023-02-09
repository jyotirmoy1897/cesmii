"""
Copyright © 2023, Abhishek Hanchate, Satish T.S. Bukkapatnam, and Smart Manufacturing Advanced Research Team, Texas A&M University, All rights reserved.

__author__ = "Abhishek Hanchate"
__copyright__ = "Copyright (C) 2023 Abhishek Hanchate"
__version__ = "1.0"
"""

import streamlit as st
from PIL import Image

def app():
    image = Image.open('intro.png')
    
    st.markdown(""" <style> .font {
    font-size:40px ; font-family: 'Cooper Black'; text-align: center; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    
    st.markdown('<p class="font">Simantha Dashboard Use-case</p>', unsafe_allow_html=True)
    
    st.image(image, use_column_width=True)
    
    st.subheader("""
    The above figure provides an example use-case for a **Asynchronous Production System Model with two machines and one finite buffers.**!
    """)
    
    st.write("Simantha is a package for simulating discrete manufacturing systems. It is designed to model asynchronous production systems with \
             finite buffers.")

    st.write("The package provides classes for the following core manufacturing objects that are used to create a system:")
    
    st.write("**Source:** Introduces raw, unprocessed parts to the system.")
    st.write("**Machine:** Continuously retrieves, processes, and relinquishes parts. May also be subject to periodic degradation, failure, and repair.")
    st.write("**Buffer:** Stores parts awaiting processing at a machine.")
    st.write("**Sink:** Collects finished parts that exit the system.")
    st.write("**Maintainer:** Repairs degrading machines according to the specified maintenance policy.")
    
    st.write('---')









