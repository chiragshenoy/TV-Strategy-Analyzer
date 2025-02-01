import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os
import math
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import db_utils
from scipy.stats import ttest_1samp
from scipy.stats import shapiro

from utils import calculate_and_display_all_metrics

def render_performance_summary():
    st.title("Select Database File for Performance Analysis")
    st.markdown("""
        - Considers the entire duration of the stock, we cannot have a specific time period
        """)

    db_folder_path = "data/perf"

    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('.db')]

    # Dropdown to select the .db file from the folder
    if db_files:
        db_file_name = st.selectbox("Select a .db file", db_files)
        
        # Construct the full path for the selected file
        db_file_path = os.path.join(db_folder_path, db_file_name)
        
        # Load and process the selected database
        perf_connection = sqlite3.connect(db_file_path)
        
        # Query the data from the selected database
        df = pd.read_sql_query("SELECT * FROM table_name", perf_connection)
        
        calculate_and_display_all_metrics(df=df, filter_column='Unnamed',target_column='AllINR', display_charts=True)

        # Close the database connection
        perf_connection.close()

    else:
        st.write("No .db files found in the folder.")