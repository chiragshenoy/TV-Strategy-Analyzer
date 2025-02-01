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

def renderDailyClosing():
    st.title("Daily Closing")
    st.markdown("""
    This page allows you to see daily closing of a scrip
    - Choose a scrip name
    """)
        
    # Query the data from the selected database
    df = db_utils.load_daily_closing_stock_data()

    scrip = st.selectbox("Select Scrip", df['scrip'].unique())

    # Filter data for the selected scrip
    filtered_df = df[df['scrip'] == scrip]

    # Convert 'time' column to datetime if it is not already in datetime format
    filtered_df['time'] = pd.to_datetime(filtered_df['time'], unit='s')  # Assuming 'time' is in epoch/Unix timestamp

    # Create a Plotly line chart for the close price vs time
    fig = px.line(filtered_df, x='time', y='close', title=f'Close Price vs Time for {scrip}')

    # Customize hover information to show the exact date and close price
    fig.update_traces(
        hovertemplate='<b>Date: </b>%{x|%Y-%m-%d %H:%M:%S}<br>' +
                      '<b>Close Price: </b>%{y}<extra></extra>'
    )

    # Update x-axis to display dates in a human-readable format
    fig.update_xaxes(
        title="Time",
        tickformat="%Y-%m-%d",  # Change this to customize the date format (e.g., "YYYY-MM-DD")
        showgrid=True,
        tickangle=45  # Optional: To rotate the tick labels for better readability
    )

    # Update y-axis title
    fig.update_yaxes(title="Close Price")

    # Display the Plotly chart
    st.plotly_chart(fig)
