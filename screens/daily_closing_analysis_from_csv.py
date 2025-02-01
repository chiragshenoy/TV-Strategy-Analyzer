import glob
import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import render_result_for_script


def render_daily_closing_analysis_from_csv():
    st.title("Daily Closing From CSV")
    st.markdown("""
    This page allows you to see daily closing of a scrip. This uses the csv files inside data/closing/daily
    - Choose a scrip name
    """)

    # Set the directory containing CSV files
    csv_directory = "data/closing/daily"

    # Get the list of CSV files
    csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
    stock_names = [os.path.basename(file).replace(".csv", "") for file in csv_files]

    db_folder_path = "data/merged_lot"

    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('.db')]

    st.title("Stock Data Visualization")

    st.sidebar.title("Filter Options")

    # Dropdown to select stock
    selected_stock = st.selectbox("Select a Stock", stock_names)
    db_file_name = st.sidebar.selectbox("Select a .db file", db_files)
    start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2020-01-01").date())
    end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2024-12-12").date())

    # If buy and sell are all 1 contracts each, make this flag as true
    simple_units_purchased_calculation = st.checkbox("Is simple contract trades?")

    if selected_stock:
        # Load the selected stock data
        file_path = os.path.join(csv_directory, f"{selected_stock}.csv")
        try:
            data = pd.read_csv(file_path)

            # Ensure columns are as expected
            if {'time', 'close'}.issubset(data.columns):
                # Convert 'time' column to datetime
                data['time'] = pd.to_datetime(data['time'], unit='s')

                filtered_data = data[
                    (data['time'] >= pd.Timestamp(start_date)) & (data['time'] <= pd.Timestamp(end_date))]

                # Plot using Plotly Express
                fig = px.line(
                    filtered_data,
                    x='time',
                    y='close',
                    title=f"{selected_stock} - Closing Prices Over Time",
                    labels={'time': 'Year', 'close': 'Closing Price'},
                    template="plotly_white",
                    hover_data={"time": "|%b %d, %Y"}  # Show exact date (e.g., Jan 01, 2021) on hover
                )

                # Customize x-axis to display years correctly
                fig.update_xaxes(
                    dtick="M12",  # Tick every 12 months
                    tickformat="%Y",  # Format as year
                    ticklabelmode="period",  # Use full-year display for labels
                    showgrid=True
                )

                # Display the interactive plot
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("The selected file does not have the required columns: time, open, close.")
        except Exception as e:
            st.error(f"An error occurred while loading the file: {e}")

    # Dropdown to select the .db file from the folder
    if db_files:

        # Construct the full path for the selected file
        db_file_path = os.path.join(db_folder_path, db_file_name)

        # Load and process the selected database
        perf_connection = sqlite3.connect(db_file_path)

        weekly_connection = sqlite3.connect("data/closing/weekly_closing.db")
        weekly_dataframe = pd.read_sql_query("SELECT * from stock_data", weekly_connection)

        # Query the data from the selected database
        data_frame = pd.read_sql_query("SELECT * FROM table_name", perf_connection)

        filtered_data = data_frame[data_frame['scrip'] == selected_stock]

        # st.table(filtered_data)

        weekly_connection = sqlite3.connect("data/closing/weekly_closing.db")
        weekly_dataframe = pd.read_sql_query("SELECT * from stock_data", weekly_connection)

        # Query the data from the selected database
        data_frame = pd.read_sql_query("SELECT * FROM table_name", perf_connection)

        analysis_results, trades_df = render_result_for_script(selected_stock, data_frame, weekly_dataframe, start_date,
                                                               end_date,
                                                               simple_units_purchased_calculation=simple_units_purchased_calculation,
                                                               show_tables=True)

        daily_prices = data
        trades = trades_df

        # Ensure required columns are present in both files
        if {'time', 'close'}.issubset(daily_prices.columns) and {'buy_datetime', 'sell_datetime', 'buy_price',
                                                                 'units_traded'}.issubset(trades.columns):
            # Convert time columns to datetime
            daily_prices['time'] = pd.to_datetime(daily_prices['time'], unit='s')
            trades['buy_datetime'] = pd.to_datetime(trades['buy_datetime'], errors='coerce')
            trades['sell_datetime'] = pd.to_datetime(trades['sell_datetime'], errors='coerce')

            # Filter daily prices within the selected date range
            daily_prices = daily_prices[(daily_prices['time'] >= pd.Timestamp(start_date)) &
                                        (daily_prices['time'] <= pd.Timestamp(end_date))]

            # Initialize a DataFrame for daily points
            date_range = pd.date_range(daily_prices['time'].min(), daily_prices['time'].max())
            points_df = pd.DataFrame({'time': date_range})

            # Calculate daily points captured for each trade
            trades_points = []
            for _, trade in trades.iterrows():
                # Define the active duration of the trade
                trade_start = max(trade['buy_datetime'], daily_prices['time'].min())
                trade_end = min(
                    trade['sell_datetime'] if pd.notnull(trade['sell_datetime']) else daily_prices['time'].max(),
                    daily_prices['time'].max())

                # Filter daily prices within the active duration
                trade_prices = daily_prices[(daily_prices['time'] >= trade_start) &
                                            (daily_prices['time'] <= trade_end)]

                # Calculate daily points captured for the trade (daily close - buy price)
                trade_prices['points'] = (trade_prices['close'] - trade['buy_price'])

                # Include realized points on the sell date
                if pd.notnull(trade['sell_datetime']):
                    trade_prices.loc[trade_prices['time'] == trade['sell_datetime'], 'points'] += (
                            (trade['sell_price'] - trade['buy_price']) * trade['units_traded']
                    )

                # Append the results
                trades_points.append(trade_prices[['time', 'points']])

            # Combine points data for all trades
            all_points = pd.concat(trades_points)

            # Aggregate daily points
            daily_points = all_points.groupby('time').sum().reset_index()

            # Merge daily points into the main points DataFrame
            points_df = points_df.merge(daily_points, on='time', how='left').fillna(0)

            # Plot daily points captured
            fig = px.bar(
                points_df,
                x='time',
                y='points',
                title="Daily Points Captured",
                labels={'time': 'Date', 'points': 'Points Captured'},
                template="plotly_white"
            )

            # Customize axes
            fig.update_xaxes(title="Date", showgrid=True)
            fig.update_yaxes(title="Points Captured")

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(
                "Ensure both files have the required columns: 'time', 'close' in daily prices, and 'buy_datetime', 'sell_datetime', 'buy_price', 'sell_price', 'units_traded' in trades.")
