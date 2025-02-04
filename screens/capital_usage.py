import math
import os
import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils


def render_capital_usage():
    db_folder_path = "data/merged_lot"

    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('.db')]
    db_files.sort(reverse=True)

    st.sidebar.title("Filter Options")
    start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2016-01-01").date())
    end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2016-04-04").date())
    db_file_name = st.sidebar.selectbox("Select a .db file", db_files)

    # Construct the full path for the selected file
    db_file_path = os.path.join(db_folder_path, db_file_name)

    # Load and process the selected database
    strategy_connection = sqlite3.connect(db_file_path)
    dataFrame = pd.read_sql_query("SELECT * FROM table_name", strategy_connection)
    dataFrame = dataFrame.drop(columns=['drawdown_inr', 'cum_profit', 'cum_profit_inr', 'runup', 'runup_inr',
                                        'drawdown', 'price_inr', 'points_captured'])

    dataFrame['buy_datetime'] = pd.to_datetime(dataFrame['buy_datetime'], errors='coerce').dt.date
    dataFrame['sell_datetime'] = pd.to_datetime(dataFrame['sell_datetime'], errors='coerce').dt.date

    # st.write(dataFrame)

    # Apply the condition and set 'sell_datetime' to None where the condition is met
    condition = (
            (dataFrame['buy_datetime'] >= start_date) &
            (dataFrame['sell_datetime'] >= end_date) &
            (dataFrame['buy_datetime'] <= end_date)
    )

    dataFrame.loc[condition, 'sell_datetime'] = None
    dataFrame.loc[condition, 'sell_price'] = None

    # Filter data within the date range or open trades
    dataFrame = dataFrame[
        (dataFrame['buy_datetime'] >= start_date) &
        (
                (dataFrame['sell_datetime'] <= end_date) |
                (
                        dataFrame['sell_datetime'].isna() &
                        (dataFrame['buy_datetime'] <= end_date)
                )
        ) |
        condition
        ]

    # st.write(dataFrame)

    for idx, row in dataFrame.iterrows():
        buy_price = row['buy_price']
        sell_price = row['sell_price']
        contracts = row['contracts']
        buy_value = buy_price * contracts
        sell_value = sell_price * contracts

        dataFrame.at[idx, 'buy_value'] = buy_value
        dataFrame.at[idx, 'sell_value'] = sell_value

        if math.isnan(row['sell_price']):
            dataFrame.at[idx, 'open'] = True
        else:
            dataFrame.at[idx, 'open'] = False

    plot_realised_pnl_over_time(dataFrame)
    plot_capital_usage_over_time(dataFrame)
    plot_unrealised_pnl_over_time(dataFrame, end_date)


def plot_capital_usage_over_time(df):
    # st.dataframe(df)

    buy_transactions = df[['buy_datetime', 'buy_value']].copy()
    buy_transactions['capital_change'] = buy_transactions['buy_value']  # Outflow when buying
    sell_transactions = df[['sell_datetime', 'sell_value']].copy()
    sell_transactions['capital_change'] = -sell_transactions['sell_value']  # Inflow when selling

    # Combine transactions
    transactions_df = pd.concat([
        buy_transactions[['buy_datetime', 'capital_change']].rename(columns={'buy_datetime': 'date'}),
        sell_transactions[['sell_datetime', 'capital_change']].rename(columns={'sell_datetime': 'date'})
    ], ignore_index=True)

    transactions_df = transactions_df.dropna()
    transactions_df = transactions_df.sort_values(by='date')
    transactions_df['cumulative_value'] = transactions_df['capital_change'].cumsum()
    # st.dataframe(transactions_df)

    # Create the figure using Plotly
    fig = go.Figure()

    # Add the cumulative PnL trace
    fig.add_trace(go.Scatter(x=transactions_df["date"], y=transactions_df["cumulative_value"], mode="lines",
                             name="Capital Change"))

    # Update layout
    fig.update_layout(
        title="Capital Change Over Time",
        xaxis_title="Date",
        yaxis_title="Capital",
        template="plotly_dark",
        hovermode="x",
    )

    st.plotly_chart(fig)


def plot_unrealised_pnl_over_time(df, end_date):
    # st.dataframe(df)

    open_positions = df[df['open'] == True]
    # st.dataframe(open_positions)

    weekly_connection = sqlite3.connect("data/closing/weekly_closing_2.db")
    weekly_dataframe = pd.read_sql_query("SELECT * from stock_data", weekly_connection)

    incremental_pnl_df = pd.DataFrame(columns=["date", "pnl", "scrip"])
    overall_pnl_df = pd.DataFrame(columns=["pnl", "scrip"])

    for idx, row in open_positions.iterrows():
        position_start_date = row['buy_datetime']
        scrip = row['scrip']
        start_time_stamp = pd.Timestamp(position_start_date)
        position_end_date = pd.Timestamp(end_date)

        mondays = pd.date_range(start=start_time_stamp, end=position_end_date, freq="W-MON")
        # st.write(scrip)

        prev_pnl = 0  # Initialize previous PnL

        for monday in mondays:
            close = utils.get_closest_close(weekly_dataframe, scrip, monday)
            total_pnl = (close - row['buy_price']) * row['contracts']

            incremental_pnl = total_pnl - prev_pnl  # Calculate week-on-week PnL
            prev_pnl = total_pnl  # Update previous PnL for next iteration
            last_pnl = total_pnl
            # st.write(f"Scrip: {scrip}, Close: {close}, Incremental PNL: {incremental_pnl}")

            incremental_pnl_df = pd.concat(
                [incremental_pnl_df, pd.DataFrame({"date": [monday], "pnl": [incremental_pnl], "scrip": [scrip]})],
                ignore_index=True)

        overall_pnl_df = pd.concat([overall_pnl_df, pd.DataFrame({"scrip": [scrip], "pnl": [last_pnl]})])

        # st.write(f"Scrip: {scrip}, Close: {close}, Total PNL: {last_pnl}")

    # Load your dataframe (assuming it's already in df)
    incremental_pnl_df["date"] = pd.to_datetime(incremental_pnl_df["date"])  # Ensure date column is in datetime format

    # Group by date and sum the pnl
    grouped_df = incremental_pnl_df.groupby("date", as_index=False)["pnl"].sum()

    grouped_df['cumulative_value'] = grouped_df['pnl'].cumsum()

    # st.dataframe(pnl_df)
    # st.dataframe(grouped_df)

    fig = go.Figure()

    # Add the cumulative PnL trace
    fig.add_trace(go.Scatter(x=grouped_df["date"], y=grouped_df["cumulative_value"], mode="lines",
                             name="Unrealised Pnl"))

    # Update layout
    fig.update_layout(
        title="Unrealised Pnl",
        xaxis_title="Date",
        yaxis_title="Unrealised PnL",
        template="plotly_dark",
        hovermode="x",
    )

    st.plotly_chart(fig)

    total_pnl = grouped_df['pnl'].sum()
    st.header("Unrealised Pnl: " + utils.format_number(total_pnl))
    st.dataframe(overall_pnl_df)


def plot_realised_pnl_over_time(df):
    # Convert datetime columns
    df["buy_datetime"] = pd.to_datetime(df["buy_datetime"])
    df["sell_datetime"] = pd.to_datetime(df["sell_datetime"])

    # Create transactions with correct cash flow signs
    buy_transactions = df[['buy_datetime', 'buy_value']].copy()
    buy_transactions['net_worth_change'] = -buy_transactions['buy_value']  # Outflow when buying
    sell_transactions = df[['sell_datetime', 'sell_value']].copy()
    sell_transactions['net_worth_change'] = +sell_transactions['sell_value']  # Inflow when selling

    # Combine transactions
    transactions_df = pd.concat([
        buy_transactions[['buy_datetime', 'net_worth_change']].rename(columns={'buy_datetime': 'date'}),
        sell_transactions[['sell_datetime', 'net_worth_change']].rename(columns={'sell_datetime': 'date'})
    ], ignore_index=True)

    # Sort and calculate cumulative net worth
    transactions_df = transactions_df.sort_values('date').reset_index(drop=True)
    transactions_df['net_worth'] = transactions_df['net_worth_change'].cumsum()

    # Prepare P&L data
    pnl_df = df.copy()
    pnl_df['pnl'] = pnl_df['sell_value'] - pnl_df['buy_value']
    pnl_df = pnl_df[['sell_datetime', 'pnl']].rename(columns={'sell_datetime': 'date'})

    # Resample net worth to weekly (forward-fill to handle weeks without transactions)
    transactions_df.set_index('date', inplace=True)
    weekly_net_worth = transactions_df.resample('W')['net_worth'].last().ffill().reset_index()
    weekly_net_worth.rename(columns={'date': 'week'}, inplace=True)

    # Aggregate P&L by week (sum with zero for missing weeks)
    pnl_df.set_index('date', inplace=True)
    weekly_pnl = pnl_df.resample('W')['pnl'].sum().reset_index()
    weekly_pnl.rename(columns={'date': 'week'}, inplace=True)

    # Create figures
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_net_worth["week"],
        y=weekly_net_worth["net_worth"],
        name="Net Worth",
        line=dict(color="blue")
    ))

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=weekly_pnl["week"],
        y=weekly_pnl["pnl"],
        name="Profit & Loss",
        marker=dict(color="green"),
        opacity=0.6
    ))

    # Streamlit App
    # st.title("Net Worth & Profit/Loss Analysis")

    merged_df = weekly_net_worth.join(weekly_pnl.set_index('week'), on='week')

    for idx, row in merged_df.iterrows():
        if math.isnan(row['pnl']):
            merged_df.at[idx, 'pnl'] = 0

    for idx, row in merged_df.iterrows():
        final = row['pnl'] + row['net_worth']
        merged_df.at[idx, 'final_pnl'] = final

    # Convert the Date column to datetime format
    merged_df["week"] = pd.to_datetime(merged_df["week"])

    # Calculate cumulative PnL
    merged_df["CumulativePnL"] = merged_df["pnl"].cumsum()

    # st.dataframe(merged_df)

    # Create the figure using Plotly
    fig = go.Figure()

    # Add the cumulative PnL trace
    fig.add_trace(go.Scatter(x=merged_df["week"], y=merged_df["CumulativePnL"], mode="lines", name="Realised PnL"))

    # Update layout
    fig.update_layout(
        title="Realised PnL Over Time",
        xaxis_title="Date",
        yaxis_title="Realised PnL",
        template="plotly_dark",
        hovermode="x",
    )

    st.plotly_chart(fig)

    total_pnl = weekly_pnl['pnl'].sum()
    st.header("Realised Pnl: " + utils.format_number(total_pnl))


def plot_nw(df):
    # Convert datetime columns
    df["buy_datetime"] = pd.to_datetime(df["buy_datetime"])
    df["sell_datetime"] = pd.to_datetime(df["sell_datetime"])
    # df = df.dropna(subset=["sell_datetime"])  # Remove missing values

    # Sort trades by date
    df = df.sort_values(["buy_datetime", "sell_datetime"])

    # Initialize tracking variables
    money_invested = 0  # Capital currently in active trades
    realized_profit = 0  # Total profit/loss from exited trades
    net_worth_records = []  # Store weekly net worth updates

    # Process each trade
    for _, row in df.iterrows():
        money_invested += row["buy_value"]  # Add capital when buying
        money_invested -= row["buy_value"]  # Remove capital when exiting
        realized_profit += (row["sell_value"] - row["buy_value"])  # Add realized profit/loss
        net_worth = money_invested + realized_profit  # Net Worth = Active Capital + Realized Profit
        net_worth_records.append((row["sell_datetime"], net_worth))  # Track updated net worth

    # Convert net worth changes to DataFrame
    net_worth_df = pd.DataFrame(net_worth_records, columns=["date", "net_worth"])

    # Aggregate by week
    net_worth_df["week"] = net_worth_df["date"].dt.to_period("W").astype(str)
    weekly_net_worth = net_worth_df.groupby("week")["net_worth"].last().reset_index()

    # Convert week period to actual dates (start of each week)
    weekly_net_worth["week"] = pd.to_datetime(weekly_net_worth["week"].str[:10])

    # Plot with Plotly
    fig = px.line(
        weekly_net_worth,
        x="week",
        y="net_worth",
        title="Weekly Net Worth Over Time",
        labels={"week": "Week", "net_worth": "Total Net Worth"},
    )

    # Streamlit App
    st.title("Net Worth Evolution")
    st.plotly_chart(fig)


# Function to aggregate data by week
def aggregate_weekly_data(data):
    # Create separate dataframes for buy and sell transactions
    buy_weekly = data.groupby(pd.Grouper(key='buy_datetime', freq='W-MON')).agg({
        'buy_value': 'sum'
    }).reset_index()

    sell_weekly = data.dropna(subset=['sell_datetime']).groupby(pd.Grouper(key='sell_datetime', freq='W-MON')).agg({
        'sell_value': 'sum'
    }).reset_index()

    # Merge buy and sell data
    weekly_data = pd.merge(buy_weekly, sell_weekly,
                           left_on='buy_datetime',
                           right_on='sell_datetime',
                           how='outer')

    # Fill NaN values with 0
    weekly_data = weekly_data.fillna(0)

    # Calculate net value (sell - buy)
    weekly_data['net_value'] = weekly_data['sell_value'] - weekly_data['buy_value']

    # Use buy_datetime as the main date column
    weekly_data['date'] = weekly_data['buy_datetime'].fillna(weekly_data['sell_datetime'])

    # Sort by date
    weekly_data = weekly_data.sort_values('date')

    return weekly_data


def plot_net_value_graph(dataFrame):
    dataFrame['buy_datetime'] = pd.to_datetime(dataFrame['buy_datetime'])
    dataFrame['sell_datetime'] = pd.to_datetime(dataFrame['sell_datetime'])

    weekly_data = aggregate_weekly_data(dataFrame)

    # Create visualization
    fig = go.Figure()

    # Add traces for buy, sell, and net values
    fig.add_trace(go.Scatter(
        x=weekly_data['date'],
        y=weekly_data['buy_value'],
        name='Buy Value',
        line=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=weekly_data['date'],
        y=weekly_data['sell_value'],
        name='Sell Value',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=weekly_data['date'],
        y=weekly_data['net_value'],
        name='Net Value',
        line=dict(color='blue')
    ))

    # Update layout
    fig.update_layout(
        title='Weekly Trading Values',
        xaxis_title='Week',
        yaxis_title='Value (₹)',
        hovermode='x unified',
        showlegend=True
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display summary statistics
    st.subheader('Summary Statistics')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Total Buy Value', f"₹{weekly_data['buy_value'].sum():,.2f}")
    with col2:
        st.metric('Total Sell Value', f"₹{weekly_data['sell_value'].sum():,.2f}")
    with col3:
        net_profit = weekly_data['net_value'].sum()
        st.metric('Net Profit/Loss', f"₹{net_profit:,.2f}",
                  delta=f"{(net_profit / weekly_data['buy_value'].sum() * 100):.2f}%")

    # Display raw data table
    st.subheader('Weekly Data')
    st.dataframe(weekly_data[['date', 'buy_value', 'sell_value', 'net_value']].style.format({
        'buy_value': '₹{:,.2f}',
        'sell_value': '₹{:,.2f}',
        'net_value': '₹{:,.2f}'
    }))
