import os
import sqlite3

import pandas as pd
import streamlit as st

import utils
from screens.capital_usage import render_capital_usage
from screens.daily_closing import render_daily_closing
from screens.performance_comparison import render_performance_comparison
from screens.performance_summary import render_performance_summary
from screens.weekly_closing import render_weekly_closing

st.set_page_config(layout="wide")

# Streamlit Sidebar for Page Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to",
                        ["Capital Usage" ,"List of Trades", "Weekly Closing", "Daily Closing from CSV", "Performance Summary",
                         "Performance Comparison",
                         "Daily Closing"])

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
    db_files.sort(reverse=True)

    # Dropdown to select the .db file from the folder
    if db_files:

        st.sidebar.title("Filter Options")
        start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2016-01-01").date())
        end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2016-04-04").date())
        db_file_name = st.sidebar.selectbox("Select a .db file", db_files)

        # Construct the full path for the selected file
        db_file_path = os.path.join(db_folder_path, db_file_name)

        # Load and process the selected database
        strategy_connection = sqlite3.connect(db_file_path)

        weeklyConnection = sqlite3.connect("data/closing/weekly_closing_2.db")
        weeklyDataframe = pd.read_sql_query("SELECT * from stock_data", weeklyConnection)

        # Query the data from the selected database
        dataFrame = pd.read_sql_query("SELECT * FROM table_name", strategy_connection)

        # Select scrip
        scrips = dataFrame['scrip'].unique()
        showTables = st.checkbox("Show trades?")

        total_realised_pnl = 0
        total_unrealised_pnl = 0

        total_number_of_stocks_traded = 0

        all_summaries = []

        for scrip in scrips:

            analysis_results, filtered_df = utils.render_result_for_script(scrip, dataFrame, weeklyDataframe,
                                                                           start_date,
                                                                           end_date,
                                                                           show_tables=showTables)

            closed_trade_count = analysis_results['trade_analysis']['closed_trades']['total_count']
            open_trade_count = analysis_results['trade_analysis']['open_trades']['total_count']

            if closed_trade_count > 0 or open_trade_count > 0:
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

        st.header("CAGR Calculation")

        initial_portfolio_value = total_number_of_stocks_traded * 100000
        final_portfolio_value = initial_portfolio_value + total_realised_pnl + total_unrealised_pnl

        st.write("Total investment Rs." + utils.format_number(initial_portfolio_value))
        st.write("Final investment Rs." + utils.format_number(final_portfolio_value))

        # Calculate CAGR
        # TODO: Divide by 0 when checking for lesser than 1 year range
        n_years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365
        cagr = (final_portfolio_value / initial_portfolio_value) ** (1 / n_years) - 1
        st.header(f"CAGR: {cagr * 100:.2f}%")

if page == "Weekly Closing":
    render_weekly_closing()

if page == "Performance Summary":
    render_performance_summary()

if page == "Performance Comparison":
    render_performance_comparison()

if page == "Daily Closing Analysis from CSV":
    pass
    # render_daily_closing_analysis_from_csv()

if page == "Daily Closing":
    render_daily_closing()

if page == "Capital Usage":
    render_capital_usage()