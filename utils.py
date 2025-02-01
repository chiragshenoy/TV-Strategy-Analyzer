from babel.numbers import format_decimal
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

def format_number(value):
    """
    Format a number:
    - Rounded to 2 decimal places.
    - Displayed in the Indian number system with commas.
    """
    if value is None or pd.isna(value):
        return "N/A"  # Handle NaN or None values gracefully
    rounded_value = round(value, 2)  # Round to 2 decimal places
    return format_decimal(rounded_value, format="#,##,##0.00", locale='en_IN')

def get_closest_close(df, scrip_name, input_time):
    """
    Get the closing price for a specific scrip at the nearest time less than the given input time.

    :param df: DataFrame containing the stock data.
    :param scrip_name: The specific stock (scrip) name to filter.
    :param input_time: The input time (in Unix timestamp format).
    :return: Closing price nearest to but less than the input time, or None if no match.
    """
    # Ensure 'time' is treated as an integer
    if df['time'].dtype != 'int64' and df['time'].dtype != 'float64':
        df['time'] = df['time'].astype(int)
    
    # Ensure input_time is also an integer
    if isinstance(input_time, pd.Timestamp):
        input_time = int(input_time.timestamp())
    elif not isinstance(input_time, (int, float)):
        input_time = int(input_time)
    
    # Filter for the given scrip
    scrip_df = df[df['scrip'] == scrip_name]
    
    # Filter for times less than the input time
    time_filtered_df = scrip_df[scrip_df['time'] < input_time]
    
    # Check if the filtered DataFrame is empty
    if time_filtered_df.empty:
        return 0  # Return None if no matching rows found
    
    # Find the row with the maximum time (nearest to the input time)
    closest_row = time_filtered_df.loc[time_filtered_df['time'].idxmax()]
    
    # Return the closing price
    return round(float(closest_row['close']), 2)


def calculate_and_display_all_metrics(df, filter_column, target_column, display_charts):
    """
    Calculate and display all metrics for unique values in a column in a single stylish table.

    :param df: DataFrame to query.
    :param filter_column: Column to filter on (e.g., 'Unnamed').
    :param target_column: Column to calculate metrics on (e.g., 'AllINR').
    """
    # Create a list to store results for the table
    metrics_data = []

    # Iterate over all unique values in the filter column
    unique_values = df[filter_column].unique()

    net_profit_df = df[df["Unnamed"] == "Profit Factor"]
    cleaned_net_profit_df = net_profit_df[['AllINR', 'scrip']].dropna().sort_values(by='AllINR')
    
    results = mathematical_plot(cleaned_net_profit_df, display_charts)

    if display_charts:
        # Create the Plotly bar chart
        fig = px.bar(
            cleaned_net_profit_df,
            x='scrip',
            y='AllINR',
            title='Profit Factor vs Scrip',
            labels={'scrip': 'Scrip', 'AllINR': 'AllINR'},
            text='AllINR',
        )

        # Customize the chart
        fig.update_traces(textposition='outside', marker_color='skyblue')
        fig.update_layout(
            xaxis_tickangle=45,
            xaxis=dict(
                tickmode='linear',
                automargin=True,
                showticklabels=True,
            ),
            margin=dict(r=20, t=50, b=100, l=50),
            height=600,
            width=1200,  # Make the graph wide enough for scrolling
        )

        # Enable horizontal scrolling
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),  # Add a range slider for interactivity
                fixedrange=False,
            )
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    for unique_value in unique_values:
        # Filter the DataFrame
        filtered_df = df[df[filter_column] == unique_value]
        
        # Group by 'scrip' and calculate mean for the target column
        grouped_metric = filtered_df.groupby('scrip')[target_column].mean()
        
        # Calculate the average and median
        average_metric = grouped_metric.mean()
        median_metric = grouped_metric.median()
        
        # Append results to the list
        metrics_data.append({
            "Metric Name": unique_value,
            "Average": round(average_metric, 2),
            "Median": round(median_metric, 2)
        })

    # Convert the list of metrics to a DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    if display_charts:
        st.markdown("## 📋 Consolidated Metrics Table", unsafe_allow_html=True)

        # Display the dataframe with larger font and styled table
        st.dataframe(metrics_df.style.format(precision=2), use_container_width=True, height = 1000)

    return results

def mathematical_plot(df, display_charts):
    # Compute metrics
    mean_pf = df['AllINR'].mean()
    median_pf = df['AllINR'].median()
    std_pf = df['AllINR'].std()
    iqr_pf = df['AllINR'].quantile(0.75) - df['AllINR'].quantile(0.25)
    skewness = df['AllINR'].skew()
    kurtosis = df['AllINR'].kurt()

    if display_charts:
        st.write("### Key Metrics of Profit Factor")
        st.write(f"Mean Profit Factor: {mean_pf:.2f}")
        st.write(f"Median Profit Factor: {median_pf:.2f}")
        st.write(f"Standard Deviation: {std_pf:.2f}")
        st.write(f"Interquartile Range (IQR): {iqr_pf:.2f}")
        st.write(f"Skewness: {skewness:.2f}")
        st.write(f"Kurtosis: {kurtosis:.2f}")

    q1, q3 = df['AllINR'].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter outliers
    filtered_df = df[(df['AllINR'] >= lower_bound) & (df['AllINR'] <= upper_bound)]

    # Recalculate metrics
    filtered_mean = filtered_df['AllINR'].mean()
    filtered_median = filtered_df['AllINR'].median()
    filtered_std = filtered_df['AllINR'].std()

    if display_charts:
        st.write(f"Filtered Mean: {filtered_mean:.2f}")
        st.write(f"Filtered Median: {filtered_median:.2f}")
        st.write(f"Filtered Standard Deviation: {filtered_std:.2f}")

        # Log-transformed Histogram
        st.write("### Log-transformed Histogram")
        df['Log_AllINR'] = np.log(df['AllINR'].replace(0, np.nan)).dropna()  # Handle zeros safely
        fig, ax = plt.subplots()
        sns.histplot(df['Log_AllINR'], kde=True, bins=30, ax=ax, color='orange')
        ax.set_title("Log-Transformed Histogram of Profit Factors")
        st.pyplot(fig)

    return {
        "mean": mean_pf,
        "median": median_pf,
        "std": std_pf,
        "iqr": iqr_pf,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "filtered_mean": filtered_mean,
        "filtered_median": filtered_median,
        "filtered_std": filtered_std,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }
