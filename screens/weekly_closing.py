import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st


def render_weekly_closing():
    st.title("Scrip Weekly Close Price Visualization")

    db_file_path = "data/closing/weekly_closing.db"

    # Load and process the selected database
    weekly_connection = sqlite3.connect(db_file_path)

    # Query the data from the selected database
    df = pd.read_sql_query("SELECT * FROM stock_data", weekly_connection)

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
    weekly_connection.close()
