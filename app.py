import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os
import datetime
import math


st.set_page_config(layout="wide")

# Streamlit Sidebar for Page Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Performance Summary", "List of Trades", "Filtered List of Trades", "Visualization", "Settings"])

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
        return None  # Return None if no matching rows found
    
    # Find the row with the maximum time (nearest to the input time)
    closest_row = time_filtered_df.loc[time_filtered_df['time'].idxmax()]
    
    # Return the closing price
    return closest_row['close']

def calculate_and_display_all_metrics(df, filter_column, target_column):
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

    st.markdown("## 📋 Consolidated Metrics Table", unsafe_allow_html=True)
    
    # Apply custom CSS to enlarge the table and enhance readability
    st.markdown("""
        <style>
            .dataframe {
                font-size: 18px;  /* Increase font size */
                border-collapse: collapse;
                width: 100%;
                height: auto;
            }
            .dataframe th, .dataframe td {
                padding: 12px;  /* Increase padding */
                text-align: center;
            }
            .dataframe th {
                background-color: #f2f2f2;  /* Header background */
            }
            .dataframe tr:nth-child(even) {
                background-color: #f9f9f9;  /* Alternating row colors */
            }
            .dataframe tr:hover {
                background-color: #f1f1f1;  /* Hover effect */
            }
        </style>
    """, unsafe_allow_html=True)

    # Display the dataframe with larger font and styled table
    st.dataframe(metrics_df.style.format(precision=2), use_container_width=True)

# List of Trades Page
if page == "List of Trades":
    st.title("List of Trades")
    st.markdown("""
    This app allows you to analyze scrip data from the SQLite database. You can:
    - See all trades
    - Filter trades for a particular scrip
    """)

    db_folder_path = "data/lot"

    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('.db')]

    # Dropdown to select the .db file from the folder
    if db_files:
        db_file_name = st.selectbox("Select a .db file", db_files)
        
        # Construct the full path for the selected file
        db_file_path = os.path.join(db_folder_path, db_file_name)
        
        # Load and process the selected database
        perfConnection = sqlite3.connect(db_file_path)
        
        # Query the data from the selected database
        df = pd.read_sql_query("SELECT * FROM table_name", perfConnection)
        
        st.dataframe(df)

        scrip = st.selectbox("Select Scrip", df['scrip'].unique())

        # Filter data for the selected scrip
        filtered_df = df[df['scrip'] == scrip]

        st.dataframe(filtered_df)

        # Close the database connection
        perfConnection.close()


# Data View Page
elif page == "Filtered List of Trades":
    
    st.title("Filtered List of Trades")
    db_folder_path = "data/lot"

    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('.db')]

    # Dropdown to select the .db file from the folder
    if db_files:
        db_file_name = st.selectbox("Select a .db file", db_files)
        
        # Construct the full path for the selected file
        db_file_path = os.path.join(db_folder_path, db_file_name)
        
        # Load and process the selected database
        perfConnection = sqlite3.connect(db_file_path)

        weeklyConnection = sqlite3.connect("data/closing/weekly_closing.db")
        weeklyDataframe = pd.read_sql_query("SELECT * from stock_data", weeklyConnection)
        # st.dataframe(weeklyDataframe)

        # Query the data from the selected database
        dataFrame = pd.read_sql_query("SELECT * FROM table_name", perfConnection)
        
        st.sidebar.title("Filter Options")

        # Select scrip
        scrips = dataFrame['scrip'].unique()
        selected_scrip = st.sidebar.selectbox("Select Scrip", scrips)

        # Filter data for the selected scrip
        filtered_data = dataFrame[dataFrame['scrip'] == selected_scrip]

        # Date range picker
        start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2021-01-01").date())
        end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2024-11-11").date())

        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        filtered_data['DateTime'] = pd.to_datetime(filtered_data['DateTime'], errors='coerce')

        # Filter trades that were opened (entries) during the date range
        opened_trades = filtered_data[
            (filtered_data['Type'].str.contains("Entry", na=False)) &
            (filtered_data['DateTime'] >= start_date) &
            (filtered_data['DateTime'] <= end_date)
        ]

        # Filter trades that were exited (exits) during the date range
        exited_trades = filtered_data[
            (filtered_data['Type'].str.contains("Exit", na=False)) &
            (filtered_data['DateTime'] >= start_date) &
            (filtered_data['DateTime'] <= end_date)
        ]

        # Ensure corresponding entry trades exist for exited trades
        valid_exited_trades = pd.merge(
            exited_trades, 
            opened_trades[['Trade', 'DateTime']],  # Only keep Trade IDs and entry DateTime
            on='Trade',
            how='inner',
            suffixes=('_exit', '_entry')
        )

        # Combine valid exits and opened trades
        combined_trades = pd.concat([opened_trades, valid_exited_trades])

        # Main page
        st.title("Filtered Trades")
        st.write(f"Displaying trades for **{selected_scrip}** from {start_date.date()} to {end_date.date()}")

        # Show filtered data
        columns_to_display = ['Trade', 'Type', 'Signal', 'DateTime', 'PriceINR','DateTime_entry', 'ProfitINR', 'Profit', 'Contracts']

        combined_trades['DateTime'] = combined_trades['DateTime'].fillna(combined_trades['DateTime_exit'])
        combined_trades = combined_trades.sort_values(by=['Trade', 'DateTime'], ascending=[False, False])

        st.dataframe(combined_trades[columns_to_display])

        # Profit calculation
        realised_profit = 0
        unrealised_profit = 0

        partsOfEntry = 3
        capital_per_trade = 100000

        # Loop through each row to calculate the profit for each trade
        # Initialize variables

        results = []  # To store the results for the table

        for _, row in combined_trades.iterrows():
            if 'Entry' in row['Type']:  # Only consider the entry for the trade
                entry_price = row['PriceINR']
                entry_contracts = row['Contracts']

                if not math.isnan(entry_contracts) and not math.isnan(entry_price):
                    units_purchased = math.floor(capital_per_trade / entry_price / (partsOfEntry / entry_contracts))  # Units
                    entry_trade_id = row['Trade']

                    # Find the corresponding exit trade
                    exit_trade = combined_trades[
                        (combined_trades['Trade'] == entry_trade_id) & 
                        (combined_trades['Type'] == 'Exit Long')
                    ]

                    if not exit_trade.empty:
                        exit_price = exit_trade.iloc[0]['PriceINR']
                        points_captured = exit_price - entry_price
                        profit = points_captured * units_purchased

                        # Append to results
                        results.append({
                            "Trade ID": entry_trade_id,
                            "Entry Price": entry_price,
                            "Exit Price": exit_price,
                            "Units Purchased": units_purchased,
                            "Points Captured": points_captured,
                            "Total PnL": profit,
                            "Status": "Closed"
                        })

                        realised_profit += profit
                # Check open position exists
                else:
                    if row['Signal'] == 'Buy':
                        closest_close = get_closest_close(weeklyDataframe, selected_scrip, end_date)
                        points_captured = float(closest_close) - entry_price

                        # Filter for the Entry Long trades (Type == 'Entry Long') and those without an Exit Long counterpart
                        entry_trades = combined_trades[(combined_trades['Type'] == 'Entry Long') & combined_trades['DateTime_entry'].isna()]

                        # Now, for each entry, count how many other entries exist with the same PriceINR and DateTime
                        entry_trades['Same_Price_DateTime_Count'] = entry_trades.groupby(['PriceINR', 'DateTime'])['PriceINR'].transform('count')

                        # Reduce the count by 1 for trades that do not have a matching trade number (Exit Long)
                        for idx, row in entry_trades.iterrows():
                            matching_exit = combined_trades[(combined_trades['Type'] == 'Exit Long') & (combined_trades['DateTime_entry'] == row['DateTime'])]
                            if matching_exit.empty:
                                entry_trades.at[idx, 'Same_Price_DateTime_Count'] -= 1

                        # Find the highest Trade number and get its corresponding count
                        highest_trade = entry_trades['Trade'].max()
                        highest_trade_count = entry_trades[entry_trades['Trade'] == highest_trade]['Same_Price_DateTime_Count'].values[0]
                        
                        # Come up with better logic for this
                        if highest_trade_count == partsOfEntry:
                            number_of_open_contracts = 1
                        elif highest_trade == 2:
                            number_of_open_contracts = 2
                        elif highest_trade_count == 0:
                            number_of_open_contracts = partsOfEntry
                        
                        units_per_contract = math.floor(capital_per_trade / entry_price / partsOfEntry)

                        st.write("Closing price: " + str(closest_close))
                        unrealised_profit = points_captured * units_per_contract * number_of_open_contracts

                        # Append to results
                        results.append({
                            "Trade ID": highest_trade,
                            "Entry Price": entry_price,
                            "Exit Price": "Holding :)",
                            "Units Purchased": units_per_contract * number_of_open_contracts,
                            "Points Captured": points_captured,
                            "Total PnL": unrealised_profit,
                            "Status": "Open"
                        })

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        # Display the table and total PnL
        st.write("### Trade Summary")
        st.dataframe(results_df)

        st.write(f"### Realised PnL: Rs. {realised_profit:.2f}")
        st.write(f"### Unrealised PnL: Rs. {unrealised_profit:.2f}")

        # Close the database connection
        perfConnection.close()

# Visualization Page
elif page == "Visualization":
    st.title("Scrip Close Price Visualization")
            
    db_file_path = "data/closing/weekly_closing.db"
        
    # Load and process the selected database
    weeklyConnection = sqlite3.connect(db_file_path)
        
    # Query the data from the selected database
    df = pd.read_sql_query("SELECT * FROM stock_data", weeklyConnection)

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
    weeklyConnection.close()

# Settings Page
elif page == "Settings":
    st.title("Settings")

elif page == "Performance Summary":
    st.title("Select Database File for Performance Analysis")
    db_folder_path = "data/perf"

    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('.db')]

    # Dropdown to select the .db file from the folder
    if db_files:
        db_file_name = st.selectbox("Select a .db file", db_files)
        
        # Construct the full path for the selected file
        db_file_path = os.path.join(db_folder_path, db_file_name)
        
        # Load and process the selected database
        perfConnection = sqlite3.connect(db_file_path)
        
        # Query the data from the selected database
        df = pd.read_sql_query("SELECT * FROM table_name", perfConnection)
        
        calculate_and_display_all_metrics(df=df, filter_column='Unnamed',target_column='AllINR')

        # Close the database connection
        perfConnection.close()

        

    else:
        st.write("No .db files found in the folder.")


