"""
Copyright © 2023, Abhishek Hanchate, Satish T.S. Bukkapatnam, and Smart Manufacturing Advanced Research Team, Texas A&M University, All rights reserved.

__author__ = "Abhishek Hanchate"
__copyright__ = "Copyright (C) 2023 Abhishek Hanchate"
__version__ = "1.0"
"""

import streamlit as st

# Custom imports 
from multipage import MultiPage
from pages import introduction, example, data_visualize # import your pages here

# Create an instance of the app 
app = MultiPage()


# Add all your applications (pages) here
app.add_page("Introduction", introduction.app)
app.add_page("Example Use-case", example.app)
app.add_page("Data Visualization",data_visualize.app)

# The main app
app.run()
