import math
import os
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import utils


# Need to add closest close

def render_weekly_closing():
    st.title("Scrip Weekly Close Price Visualization")
    st.sidebar.title("Filter Options")
    start_date = pd.to_datetime(st.sidebar.date_input("Start Date", value=pd.Timestamp("2016-01-01").date())).date()
    end_date = pd.to_datetime(st.sidebar.date_input("End Date", value=pd.Timestamp("2020-01-01").date())).date()

    db_strategies_path = "data/merged_lot"

    db_files = [f for f in os.listdir(db_strategies_path) if f.endswith('.db')]
    db_files.sort(reverse=True)

    st.sidebar.title("Filter Options")
    db_file_name = st.sidebar.selectbox("Select a .db file", db_files)

    # Construct the full path for the selected file
    db_file_path = os.path.join(db_strategies_path, db_file_name)

    # Load and process the selected database
    trades_connection = sqlite3.connect(db_file_path)
    weeklyDataframe = pd.read_sql_query("SELECT * from table_name", trades_connection)

    db_file_path = "data/closing/weekly_closing.db"

    # Load and process the selected database
    weekly_closing_connection = sqlite3.connect(db_file_path)

    # Query the data from the selected database
    df = pd.read_sql_query("SELECT * FROM stock_data", weekly_closing_connection)
    weekly_closing_connection.close()

    scrip = st.selectbox("Select Scrip", df['scrip'].unique())

    # Filter data for the selected scrip
    closing_prices_df = df[df['scrip'] == scrip]

    # st.dataframe(closing_prices_df)

    # Convert 'time' column to datetime if it is not already in datetime format
    closing_prices_df['time'] = pd.to_datetime(closing_prices_df['time'],
                                               unit='s')  # Assuming 'time' is in epoch/Unix timestamp

    closing_prices_df = closing_prices_df[
        (closing_prices_df['time'].dt.date >= start_date) &
        (closing_prices_df['time'].dt.date <= end_date)
        ]

    first_closing_price = float(closing_prices_df.iloc[0]['close'])
    last_closing_price = float(closing_prices_df.iloc[-1]['close'])
    diff_closing_price = last_closing_price - first_closing_price
    percentage_diff_closing_price = (diff_closing_price / first_closing_price) * 100

    # Create a Plotly line chart for the close price vs time
    fig = px.line(closing_prices_df, x='time', y='close',
                  title=f'Close Price vs Time for {scrip}. Buy and hold gains {percentage_diff_closing_price:.2f}%')

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
        tickangle=0  # Optional: To rotate the tick labels for better readability
    )

    # Update y-axis title
    fig.update_yaxes(title="Close Price")

    # Display the Plotly chart
    st.plotly_chart(fig)

    # PnL Section

    trades_df = weeklyDataframe[weeklyDataframe['scrip'] == scrip]
    trades_df = trades_df.drop(columns=['drawdown_inr', 'cum_profit', 'cum_profit_inr', 'runup', 'runup_inr',
                                        'drawdown', 'scrip', 'price_inr', 'points_captured'])

    trades_df['buy_datetime'] = pd.to_datetime(trades_df['buy_datetime'], errors='coerce').dt.date
    trades_df['sell_datetime'] = pd.to_datetime(trades_df['sell_datetime'], errors='coerce').dt.date

    # Filter out trades which are closing after the end date, find the last price at the end date,
    # Make that trade as an open trade

    # Apply the condition and set 'sell_datetime' to None where the condition is met
    condition = (
            (trades_df['buy_datetime'] >= start_date) &
            (trades_df['sell_datetime'] >= end_date) &
            (trades_df['buy_datetime'] <= end_date)
    )

    trades_df.loc[condition, 'sell_datetime'] = None
    trades_df.loc[condition, 'sell_price'] = None

    # Filter data within the date range or open trades
    trades_df = trades_df[
        (trades_df['buy_datetime'] >= start_date) &
        (
                (trades_df['sell_datetime'] <= end_date) |
                (
                        trades_df['sell_datetime'].isna() &
                        (trades_df['buy_datetime'] <= end_date)
                )
        ) |
        condition
        ]

    # Handle open position. Objective is to find out contracts of open position
    initial_capital = 100000
    unrealised_profit = 0

    # Check if there's an open position
    if not trades_df[trades_df["sell_datetime"].isna()].empty:
        # Get the buy_datetime and buy_price of the open position
        open_trade = trades_df[trades_df["sell_datetime"].isna()].iloc[0]
        open_buy_datetime = open_trade["buy_datetime"]
        open_buy_price = open_trade["buy_price"]

        # Find matched trades (same buy_datetime and buy_price) that are NOT open
        matched_trades = trades_df[
            (trades_df["buy_datetime"] == open_buy_datetime) &
            (trades_df["buy_price"] == open_buy_price) &
            (trades_df["sell_datetime"].notna())  # Exclude open positions
            ]

        # Sum of contracts in matched trades
        exited_contracts = matched_trades["contracts"].sum()
        initial_contracts = math.floor(initial_capital / open_buy_price)
        remaining_contracts = initial_contracts - exited_contracts
        trades_df.loc[trades_df["sell_datetime"].isna(), "contracts"] = remaining_contracts
        unrealised_profit = (last_closing_price - open_buy_price) * remaining_contracts

    # End Handle open position

    for idx, row in trades_df.iterrows():
        buy_price = row['buy_price']
        sell_price = row['sell_price']
        contracts = row['contracts']
        pnl = (sell_price - buy_price) * contracts
        trades_df.at[idx, 'pnl'] = pnl

    # Convert 'time' column to datetime
    trades_df['buy_datetime'] = pd.to_datetime(trades_df['buy_datetime']).dt.date

    closing_prices_df['time'] = pd.to_datetime(closing_prices_df['time'], unit='s').dt.date
    closing_prices_df = closing_prices_df[(closing_prices_df['time'] >= (start_date)) &
                                          (closing_prices_df['time'] <= (end_date))]

    closing_prices_df = closing_prices_df.drop(columns=['open', 'high', 'low', 'Volume', 'scrip'])
    closing_prices_df['close'] = pd.to_numeric(closing_prices_df['close'], errors='coerce')

    # Plotting

    # Create a list to store the weekly data and the cumulative realized P/L
    weekly_data = []
    realized_pl = 0  # Start with zero realized P/L

    # Iterate over each weekly closing date
    for _, row in closing_prices_df.iterrows():
        week_date = row['time']
        week_close_price = row['close']

        # Find open trades on this date
        open_trades = trades_df[(trades_df['buy_datetime'] <= week_date) &
                                ((trades_df['sell_datetime'].isna()) | (trades_df['sell_datetime'] > week_date))]

        # Compute unrealized profit/loss for open trades
        unrealized_pl = np.sum((week_close_price - open_trades['buy_price']) * open_trades['contracts'].fillna(0))

        # Compute total open contracts
        total_open_contracts = open_trades['contracts'].fillna(0).sum()

        # Process closed trades (Realized P/L)
        closed_trades = trades_df[trades_df['sell_datetime'] == week_date]

        for _, trade in closed_trades.iterrows():
            trade_pl = (trade['sell_price'] - trade['buy_price']) * trade['contracts']
            realized_pl += trade_pl  # Accumulate realized P/L

            # Store the cumulative realized P/L, total P/L, and open contracts for the week
        weekly_data.append([week_date, unrealized_pl + realized_pl, realized_pl, total_open_contracts])

    # Convert the weekly data into a DataFrame for plotting
    pl_df = pd.DataFrame(weekly_data, columns=['Date', 'Total P/L', 'Cumulative Realized P/L', 'Open Contracts'])

    # st.dataframe(pl_df)
    # Streamlit App
    st.title("Strategy's Weekly Profit/Loss Over Time")

    # Plot using Plotly with hover tooltip
    fig = px.line(pl_df, x='Date', y='Total P/L',
                  title="Weekly Total Profit/Loss (Realized + Unrealized)",
                  hover_data={"Date": "|%Y-%m-%d", "Open Contracts": True})

    # Add horizontal zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="green")
    # Update x-axis to display dates in a human-readable format
    fig.update_xaxes(
        title="Time",
        tickformat="%Y-%m-%d",  # Change this to customize the date format (e.g., "YYYY-MM-DD")
        showgrid=True,
        tickangle=0  # Optional: To rotate the tick labels for better readability
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pl_df, use_container_width=True)

    st.dataframe(trades_df, use_container_width=True)
    st.write("Total Realised PnL of strategy " + utils.format_number(trades_df['pnl'].sum()))
    st.write("Total Unrealised PnL of strategy " + utils.format_number(unrealised_profit))

    # TODO: MDD Calculation
    # max_draw_down = pl_df['Total P/L'].min()
    # st.write(f"Max Drawdown Rs. " + utils.format_number(max_draw_down) + "/-" )
