import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os
import math
import glob

st.set_page_config(layout="wide")

# Streamlit Sidebar for Page Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Merged List of Trades", "Daily Closing Analysis", "Performance Summary", "Weekly Closing", "Filtered List of Trades [Deprecated]", "Analysis of Trades [Deprecated]", "Raw List of Trades [Deprecated]"])

def renderResultForScript(scrip, dataFrame, weeklyDataframe, start_date, end_date, show_summary_trades , show_detailed_trades):
    # Filter data for the selected scrip
        filtered_data = dataFrame[dataFrame['scrip'] == scrip]

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

        if len(combined_trades) != 0:

            # Main page
            if show_detailed_trades:
                st.subheader(f"Displaying trades for **{scrip}** from {start_date.date()} to {end_date.date()}")

            # Show filtered data
            columns_to_display = ['Trade', 'Type', 'Signal', 'DateTime', 'PriceINR','DateTime_entry', 'ProfitINR', 'Profit', 'Contracts']

            combined_trades['DateTime'] = combined_trades['DateTime'].fillna(combined_trades['DateTime_exit'])
            combined_trades = combined_trades.sort_values(by=['Trade', 'DateTime'], ascending=[False, False])

            if show_detailed_trades:
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
                        if row['Signal'] == 'Buy' and not math.isnan(row['PriceINR']):
                            closest_close = get_closest_close(weeklyDataframe, scrip, end_date)
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
                            elif highest_trade_count == 2:
                                number_of_open_contracts = 2
                            elif highest_trade_count == 0:
                                number_of_open_contracts = partsOfEntry
                            
                            units_per_contract = math.floor(capital_per_trade / entry_price / partsOfEntry)

                            if show_summary_trades:
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
            if show_summary_trades:
                st.write("### Trade Summary")
                st.dataframe(results_df)

                st.write(f"### Realised PnL: Rs. {realised_profit:.2f}")
                st.write(f"### Unrealised PnL: Rs. {unrealised_profit:.2f}")

            # Close the database connection
            perfConnection.close()

            if realised_profit is None:
                realised_profit = 0
            if unrealised_profit is None:
                unrealised_profit = 0

            return (realised_profit, unrealised_profit)

        else:
            if show_summary_trades:
                st.write("No Trades found for " + scrip)

            return (0,0)

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

    # Display the dataframe with larger font and styled table
    st.dataframe(metrics_df.style.format(precision=2), use_container_width=True, height = 1000)

def renderResultsForMergedScript(scrip, dataFrame, weeklyDataframe, start_date, end_date):
    
    initial_capital = 100000
    profitBookingCount = 3
    
    # Filter data for the specific scrip
    filtered_data = dataFrame[dataFrame['scrip'] == scrip]
    filtered_data = filtered_data.drop(columns=['drawdown_inr', 'cum_profit', 'cum_profit_inr', 'runup', 'runup_inr',
                                                'drawdown', 'scrip', 'price_inr', 'points_captured'])

    # Convert start_date and end_date to Timestamps
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Ensure datetime columns are properly converted
    filtered_data.loc[:, 'buy_datetime'] = pd.to_datetime(filtered_data['buy_datetime'], errors='coerce')
    filtered_data.loc[:, 'sell_datetime'] = pd.to_datetime(filtered_data['sell_datetime'], errors='coerce')

    # Filter data within the date range or open trades
    filtered_data = filtered_data[
        (filtered_data['buy_datetime'] >= start_date) &
        ((filtered_data['sell_datetime'] <= end_date) | filtered_data['sell_datetime'].isna())
    ]

    # Process each trade
    for index, row in filtered_data.iterrows():
        
        # Case of open trade
        if math.isnan(row['contracts']):
            # Find the latest trade and calculate matching trades
            latest_trade = filtered_data.loc[filtered_data['trade_id'].idxmax()]
            latest_buy_price = latest_trade['buy_price']
            latest_buy_datetime = latest_trade['buy_datetime']

            matching_trades = len(filtered_data[
                (filtered_data['buy_price'] == latest_buy_price) &
                (filtered_data['buy_datetime'] == latest_buy_datetime)
            ])

            # Determine the number of open contracts
            if matching_trades == profitBookingCount:
                number_of_open_contracts = 1
            elif matching_trades == 2:
                number_of_open_contracts = 2
            else:
                number_of_open_contracts = profitBookingCount

            # Error handling
            if math.isnan(number_of_open_contracts):
                number_of_open_contracts = 1
            if math.isnan(latest_buy_price):
                latest_buy_price = 1

            units_per_contract = math.floor(initial_capital / latest_buy_price / profitBookingCount)

            if math.isnan(units_per_contract):
                units_per_contract = 1
            
            filtered_data.loc[index, 'units_traded'] = units_per_contract * number_of_open_contracts

        else:
            # Calculate units traded for specified contracts
            buy_price = row['buy_price']
            sell_price = row['sell_price']

            if not math.isnan(row['buy_price']):
                units_purchased = math.floor(initial_capital / row['buy_price'] / (profitBookingCount / row['contracts']))
            else:
                units_purchased = 0

            filtered_data.loc[index, 'units_traded'] = units_purchased

            if not math.isnan(row['buy_price']) and not math.isnan(row['sell_price']):
                filtered_data.loc[index, 'points_realised'] = sell_price - buy_price
                filtered_data.loc[index, 'pnl'] = (sell_price - buy_price) * units_purchased

    st.subheader(f"Displaying trades for **{scrip}** from {start_date.date()} to {end_date.date()}")

    # Sort and display data
    filtered_data = filtered_data.sort_values(by=['trade_id'], ascending=[False])
    st.dataframe(filtered_data)

    # Separate closed and open trades
    closed_trades = filtered_data[filtered_data['sell_datetime'].notna()]
    open_trades = filtered_data[filtered_data['sell_datetime'].isna()]

    open_trades = open_trades.copy()
    closed_trades = closed_trades.copy()

    closest_close = round(get_closest_close(weeklyDataframe, scrip, end_date), 2)
    st.write("Closing Price: " + str(closest_close))

    # Calculate realized metrics
    if len(closed_trades) > 0:

        realized_profit_loss = closed_trades.apply(
            lambda row: round(row['sell_price'] * row['units_traded'] if not math.isnan(row['units_traded']) else 1, 2), axis=1
        )

        # Assign the computed values to the DataFrame
        closed_trades.loc[:, 'realized_profit_loss'] = realized_profit_loss

        total_realized_profit_loss = 0

        if 'pnl' in closed_trades.columns:
            total_realized_profit_loss = closed_trades['pnl'].sum()

        realized_metrics = {
            'total_realized_profit_loss': total_realized_profit_loss,
            'total_closed_trades': len(closed_trades),
            'profitable_closed_trades': len(closed_trades[closed_trades['realized_profit_loss'] > 0]),
            'unprofitable_closed_trades': len(closed_trades[closed_trades['realized_profit_loss'] <= 0]),
            'win_rate': len(closed_trades[closed_trades['realized_profit_loss'] > 0]) / len(closed_trades) * 100
        }
    else:
        realized_metrics = {
            'total_realized_profit_loss': 0,
            'total_closed_trades': 0,
            'profitable_closed_trades': 0,
            'unprofitable_closed_trades': 0,
            'win_rate': 0
        }

    # Calculate unrealized metrics
    if len(open_trades) > 0:
        open_trades.loc[:, 'current_market_price'] = closest_close
        
        # Compute the 'unrealized_profit_loss' values first
        unrealized_profit_loss = open_trades.apply(
            lambda row: round((closest_close - row['buy_price']) * row['units_traded'], 2), axis=1
        )

        # Assign the computed values to the DataFrame
        open_trades.loc[:, 'unrealized_profit_loss'] = unrealized_profit_loss
        
        unrealized_metrics = {
            'total_unrealized_profit_loss': open_trades['unrealized_profit_loss'].sum(),
            'total_open_trades': len(open_trades)
        }
    else:
        unrealized_metrics = {
            'total_unrealized_profit_loss': 0,
            'total_open_trades': 0
        }

    # Portfolio summary
    portfolio_summary = {
        'scrip': scrip,
        'total_realized_profit_loss': round(realized_metrics['total_realized_profit_loss'], 2),
        'total_unrealized_profit_loss': round(unrealized_metrics['total_unrealized_profit_loss'], 2),
        'total_profit_loss': round(realized_metrics['total_realized_profit_loss'] + unrealized_metrics['total_unrealized_profit_loss'], 2),
        'current_portfolio_value': round(initial_capital + realized_metrics['total_realized_profit_loss'] + unrealized_metrics['total_unrealized_profit_loss'], 2)
    }

    # Detailed trade analysis
    trade_analysis = {
        'closed_trades': {
            'total_count': realized_metrics['total_closed_trades'],
            'profitable_count': realized_metrics['profitable_closed_trades'],
            'unprofitable_count': realized_metrics['unprofitable_closed_trades'],
            'win_rate': realized_metrics['win_rate']
        },
        'open_trades': {
            'total_count': unrealized_metrics['total_open_trades']
        }
    }

    jsonResult = {
        'portfolio_summary': portfolio_summary,
        'trade_analysis': trade_analysis
    }

    return jsonResult, filtered_data

# Cache database connection
@st.cache_resource
def get_daily_closing_database_connection():
    """
    Create and cache database connection.
    This prevents creating new connections on every rerun.
    """
    return sqlite3.connect('data/closing/daily_closing.db', check_same_thread=False)

# Cache data loading
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_daily_closing_stock_data():
    """
    Load and cache stock data for a specific symbol.
    TTL ensures data is refreshed periodically.
    """
    conn = get_daily_closing_database_connection()
    df = pd.read_sql_query("SELECT * FROM stock_data", conn)
    return df

if page == "Daily Closing Analysis":
    st.title("Daily Closing")
    st.markdown("""
    This page allows you to see daily closing of a scrip. This uses the csv files inside data/closing/daily
    - Choose a scrip name
    """)

    # Set the directory containing CSV files
    csv_directory = "data/closing/daily"

    # Get the list of CSV files
    csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
    stock_names = [os.path.basename(file).replace(".csv", "") for file in csv_files]

    st.title("Stock Data Visualization")
    start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2020-01-01").date())
    end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2024-12-12").date())

    # Dropdown to select stock
    selected_stock = st.selectbox("Select a Stock", stock_names)
    
    if selected_stock:
        # Load the selected stock data
        file_path = os.path.join(csv_directory, f"{selected_stock}.csv")
        try:
            data = pd.read_csv(file_path)

            # Ensure columns are as expected
            if {'time', 'close'}.issubset(data.columns):
                # Convert 'time' column to datetime
                data['time'] = pd.to_datetime(data['time'], unit='s')

                filtered_data = data[(data['time'] >= pd.Timestamp(start_date)) & (data['time'] <= pd.Timestamp(end_date))]

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

    db_folder_path = "data/merged_lot"

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

        # Query the data from the selected database
        dataFrame = pd.read_sql_query("SELECT * FROM table_name", perfConnection)
    
        filtered_data = dataFrame[dataFrame['scrip'] == selected_stock]

        # st.table(filtered_data)

        weeklyConnection = sqlite3.connect("data/closing/weekly_closing.db")
        weeklyDataframe = pd.read_sql_query("SELECT * from stock_data", weeklyConnection)

        # Query the data from the selected database
        dataFrame = pd.read_sql_query("SELECT * FROM table_name", perfConnection)
        
        st.sidebar.title("Filter Options")

        analysis_results, trades_df = renderResultsForMergedScript(selected_stock, dataFrame, weeklyDataframe, start_date,
                                                                end_date)

        daily_prices = data
        trades = trades_df

          # Ensure required columns are present in both files
        if {'time', 'close'}.issubset(daily_prices.columns) and {'buy_datetime', 'sell_datetime', 'buy_price', 'units_traded'}.issubset(trades.columns):
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
                trade_end = min(trade['sell_datetime'] if pd.notnull(trade['sell_datetime']) else daily_prices['time'].max(), daily_prices['time'].max())

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
            st.error("Ensure both files have the required columns: 'time', 'close' in daily prices, and 'buy_datetime', 'sell_datetime', 'buy_price', 'sell_price', 'units_traded' in trades.")

if page == "Daily Merged List of Trades":
    st.title("Daily Merged List of Trades")
    st.markdown("""
    This page allows you to see daily closing of a scrip
    - Choose a scrip name
    """)
        
    # Query the data from the selected database
    df = load_daily_closing_stock_data()

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

if page == "Merged List of Trades":
    st.title("Merged List of Trades")
    st.markdown("""
    This page allows you to analyze scrip data from the SQLite database. You can:
    - Choose a time period
    - Get the overall PnL status of the strategy
    - Rs.1,00,000 is the capital for each trade
    - Works for ANY time frame, as long as you have the list of trades
    """)

    db_folder_path = "data/merged_lot"

    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('.db')]

    # show_header = st.radio( "Do you want to display the header?",("Yes", "No"))

    # Dropdown to select the .db file from the folder
    if db_files:
        db_file_name = st.selectbox("Select a .db file", db_files)
        
        # Construct the full path for the selected file
        db_file_path = os.path.join(db_folder_path, db_file_name)
        
        # Load and process the selected database
        perfConnection = sqlite3.connect(db_file_path)

        weeklyConnection = sqlite3.connect("data/closing/weekly_closing.db")
        weeklyDataframe = pd.read_sql_query("SELECT * from stock_data", weeklyConnection)

        # Query the data from the selected database
        dataFrame = pd.read_sql_query("SELECT * FROM table_name", perfConnection)
        
        st.sidebar.title("Filter Options")

        # Select scrip
        scrips = dataFrame['scrip'].unique()

        start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2020-01-01").date())
        end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2024-12-12").date())

        total_realised_pnl = 0
        total_unrealised_pnl = 0

        for scrip in scrips:
            
            analysis_results, filtered_df = renderResultsForMergedScript(scrip, dataFrame, weeklyDataframe, start_date,
                                                                end_date)
            
            # for key, value in analysis_results['portfolio_summary'].items():
            #     st.write(f"{key}: {value}")

            # for key, value in analysis_results['trade_analysis'].items():
            #     st.write(f"{key}: {value}")

            portfolio_summary = analysis_results['portfolio_summary']
            
            total_realised_pnl += analysis_results['portfolio_summary']['total_realized_profit_loss']
            total_unrealised_pnl += analysis_results['portfolio_summary']['total_unrealized_profit_loss']

            # trade_analysis = analysis_results['trade_analysis']
     
            # Prepare a flattened table for Portfolio Summary
            portfolio_summary_df = pd.DataFrame.from_dict([portfolio_summary])

            # Prepare a flattened table for Trade Analysis

            # trade_analysis_data = {
            #     'Metric': ['Total Closed Trades', 'Profitable Trades', 'Unprofitable Trades', 'Win Rate', 'Open Trades'],
            #     'Value': [
            #         trade_analysis['closed_trades']['total_count'],
            #         trade_analysis['closed_trades']['profitable_count'],
            #         trade_analysis['closed_trades']['unprofitable_count'],
            #         f"{trade_analysis['closed_trades']['win_rate']}%",
            #         trade_analysis['open_trades']['total_count']
            #     ]
            # }

            # Display the tables in Streamlit
            st.subheader("Trades Summary")
            st.table(portfolio_summary_df)
            st.write("----")

            # trade_analysis_df = pd.DataFrame(trade_analysis_data)
            # st.subheader("Trade Analysis")
            # st.table(trade_analysis_df)
        
        st.header("Total Realised PnL: " + str(total_realised_pnl))
        st.header("Total Unealised PnL: " + str(total_unrealised_pnl))

if page == "Analysis of Trades [Deprecated]":
    st.title("Analysis of Trades [Deprecated]")
    st.markdown("""
    This page allows you to analyze scrip data from the SQLite database. You can:
    - Choose a time period
    - Get the overall PnL status of the strategy
    - Accurate, but uses old way of calculating PnL, without merging the trades
    """)

    db_folder_path = "data/lot"

    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('.db') and not f.startswith('merged')]

    # Dropdown to select the .db file from the folder
    if db_files:
        db_file_name = st.selectbox("Select a .db file", db_files)
        
        # Construct the full path for the selected file
        db_file_path = os.path.join(db_folder_path, db_file_name)
        
        # Load and process the selected database
        perfConnection = sqlite3.connect(db_file_path)

        weeklyConnection = sqlite3.connect("data/closing/weekly_closing.db")
        weeklyDataframe = pd.read_sql_query("SELECT * from stock_data", weeklyConnection)

        # Query the data from the selected database
        dataFrame = pd.read_sql_query("SELECT * FROM table_name", perfConnection)
        
        st.sidebar.title("Filter Options")

        # Select scrip
        scrips = dataFrame['scrip'].unique()

        start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2024-01-01").date())
        end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2024-12-12").date())

        total_realised_pnl = 0
        total_unrealised_pnl = 0

        for scrip in scrips:
            realised_pnl, unrealise_pnl = renderResultForScript(scrip, dataFrame, weeklyDataframe, start_date,
                                                                end_date,show_summary_trades = False, show_detailed_trades= False)
            total_realised_pnl += realised_pnl
            total_unrealised_pnl += unrealise_pnl

        st.header("Total realised PnL: " + str(total_realised_pnl))
        st.header("Total unrealised PnL: " + str(total_unrealised_pnl))

if page == "Raw List of Trades [Deprecated]":
    st.title("Raw List of Trades")
    st.markdown("""
    This page allows you to analyze scrip data from the SQLite database. You can:
    - See all trades
    - Filter trades for a particular scrip
    """)

    db_folder_path = "data/lot"

    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('.db') and not f.startswith('merged')]

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

if page == "Filtered List of Trades [Deprecated]":
    
    st.title("Filtered List of Trades")

    st.markdown("""
    This page allows you to choose a time range and a script.
    - Get unrealised and realised summary
    - Get trades summary
    - Works for any Time Frame
    """)

    db_folder_path = "data/lot"

    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('.db') and not f.startswith('merged')]

    # Dropdown to select the .db file from the folder
    if db_files:
        db_file_name = st.selectbox("Select a .db file", db_files)
        
        # Construct the full path for the selected file
        db_file_path = os.path.join(db_folder_path, db_file_name)
        
        # Load and process the selected database
        perfConnection = sqlite3.connect(db_file_path)

        weeklyConnection = sqlite3.connect("data/closing/weekly_closing.db")
        weeklyDataframe = pd.read_sql_query("SELECT * from stock_data", weeklyConnection)

        # Query the data from the selected database
        dataFrame = pd.read_sql_query("SELECT * FROM table_name", perfConnection)
        
        st.sidebar.title("Filter Options")

        # Select scrip
        scrips = dataFrame['scrip'].unique()
        selected_scrip = st.sidebar.selectbox("Select Scrip", scrips)

        # Date range picker
        start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2020-01-01").date())
        end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2024-12-12").date())

        renderResultForScript(selected_scrip, dataFrame, weeklyDataframe, start_date, end_date, show_summary_trades = True, show_detailed_trades = True)
        
if page == "Weekly Closing":
    st.title("Scrip Weekly Close Price Visualization")
            
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

if page == "Performance Summary":
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
        perfConnection = sqlite3.connect(db_file_path)
        
        # Query the data from the selected database
        df = pd.read_sql_query("SELECT * FROM table_name", perfConnection)
        
        calculate_and_display_all_metrics(df=df, filter_column='Unnamed',target_column='AllINR')

        # Close the database connection
        perfConnection.close()

    else:
        st.write("No .db files found in the folder.")