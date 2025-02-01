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
from screens.daily_closing import renderDailyClosing
from screens.daily_closing_analysis_from_csv import renderDailyClosingAnalysisFromCsv
import utils

from scipy.stats import ttest_1samp
from scipy.stats import shapiro

from screens.performance_comparison import renderPerformanceComparison
from screens.performance_summary import renderPerformanceSummary

from screens.weekly_closing import renderWeeklyClosing

st.set_page_config(layout="wide")

# Streamlit Sidebar for Page Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["List of Trades", "Daily Closing from CSV", "Performance Summary", "Performance Comparison", "Weekly Closing", "Daily Closing"])

def renderResultsForScript(scrip, dataFrame, weeklyDataframe, start_date, end_date, simple_units_purchased_calculation, showTables):
    
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
        (
            (filtered_data['sell_datetime'] <= end_date) | 
            (
                filtered_data['sell_datetime'].isna() & 
                (filtered_data['buy_datetime'] <= end_date)
            )
        ) | 
        ( (filtered_data['buy_datetime'] >= start_date) & (filtered_data['sell_datetime'] >= end_date) & (filtered_data['buy_datetime'] <= end_date))
    ]

    # Update 'contracts' field to NaN based on the condition
    filtered_data.loc[
        (filtered_data['buy_datetime'] >= start_date) & 
        (filtered_data['buy_datetime'] <= end_date) &
        (filtered_data['sell_datetime'] >= end_date), 
        ['contracts', 'sell_datetime', 'sell_price']
    ] = np.nan

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
                # if simple_units_purchased_calculation:
                #     units_purchased = math.floor(initial_capital / row['buy_price'])
                # else:
                #     units_purchased = math.floor(initial_capital / row['buy_price'])
                units_purchased = row['contracts']
            else:
                units_purchased = 0

            filtered_data.loc[index, 'units_traded'] = units_purchased

            if not math.isnan(row['buy_price']) and not math.isnan(row['sell_price']):
                filtered_data.loc[index, 'points_realised'] = sell_price - buy_price
                filtered_data.loc[index, 'pnl'] = (sell_price - buy_price) * units_purchased

    if showTables:
        st.subheader(f"Displaying trades for **{scrip}** from {start_date.date()} to {end_date.date()}")

    # Sort and display data
    filtered_data = filtered_data.sort_values(by=['trade_id'], ascending=[False])
    
    if showTables:
        st.dataframe(filtered_data)

    # Separate closed and open trades
    closed_trades = filtered_data[filtered_data['sell_datetime'].notna()]
    open_trades = filtered_data[filtered_data['sell_datetime'].isna()]

    open_trades = open_trades.copy()
    closed_trades = closed_trades.copy()

    closest_close = round(utils.get_closest_close(weeklyDataframe, scrip, end_date), 2)
    
    if showTables:
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

if page == "List of Trades":
    st.title("List of Trades")
    st.markdown("""
    This page allows you to analyze scrip data from the SQLite database. You can:
    - Choose a time period
    - Get the overall PnL status of the strategy
    - Rs.1,00,000 is the capital for each trade
    - Works for ANY time frame, as long as you have the list of trades
    """)

    db_folder_path = "data/merged_lot"

    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('.db')]
    db_files.sort()

    # show_header = st.radio( "Do you want to display the header?",("Yes", "No"))

    # Dropdown to select the .db file from the folder
    if db_files:
        st.sidebar.title("Filter Options")
        db_file_name = st.sidebar.selectbox("Select a .db file", db_files)
        
        # Construct the full path for the selected file
        db_file_path = os.path.join(db_folder_path, db_file_name)
        
        # Load and process the selected database
        perfConnection = sqlite3.connect(db_file_path)

        weeklyConnection = sqlite3.connect("data/closing/weekly_closing.db")
        weeklyDataframe = pd.read_sql_query("SELECT * from stock_data", weeklyConnection)

        # Query the data from the selected database
        dataFrame = pd.read_sql_query("SELECT * FROM table_name", perfConnection)

        # Select scrip
        scrips = dataFrame['scrip'].unique()

        start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2016-01-01").date())
        end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2020-01-01").date())
        
        simple_units_purchased_calculation = st.checkbox("Is simple contract trades?")
        showTables = st.checkbox("Show trades?")

        total_realised_pnl = 0
        total_unrealised_pnl = 0

        total_number_of_stocks_traded = 0

        all_summaries = []

        for scrip in scrips:
            
            analysis_results, filtered_df = renderResultsForScript(scrip, dataFrame, weeklyDataframe, start_date,
                                                                end_date, simple_units_purchased_calculation, showTables = showTables)
            
            closed_trade_count = analysis_results['trade_analysis']['closed_trades']['total_count']
            open_trade_count = analysis_results['trade_analysis']['open_trades']['total_count']
        
            if (closed_trade_count > 0 or open_trade_count > 0 ):
                total_number_of_stocks_traded = total_number_of_stocks_traded + 1

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
            if showTables:
                st.subheader("Trades Summary")
                st.table(portfolio_summary_df)
                st.write("----")

            all_summaries.append(portfolio_summary)

            # trade_analysis_df = pd.DataFrame(trade_analysis_data)
            # st.subheader("Trade Analysis")
            # st.table(trade_analysis_df)
        
        st.dataframe(all_summaries)

        st.header("Total Realised PnL: " + utils.format_number(total_realised_pnl))
        st.header("Total Unealised PnL: " + utils.format_number(total_unrealised_pnl))
        st.header("Total Stocks Traded: " + utils.format_number(total_number_of_stocks_traded))
        
if page == "Weekly Closing":
    renderWeeklyClosing()

if page == "Performance Summary":
    renderPerformanceSummary()

if page == "Performance Comparison":
    renderPerformanceComparison()

if page == "Daily Closing Analysis from CSV":
    renderDailyClosingAnalysisFromCsv()

if page == "Daily Closing":
    renderDailyClosing()