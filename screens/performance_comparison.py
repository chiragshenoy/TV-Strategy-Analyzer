import os
import sqlite3

import pandas as pd
import streamlit as st

import utils


def render_performance_comparison():
    st.title("Compare performance summaries of all strategies")
    db_folder_path = "data/perf"

    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('.db')]

    all_metrics = []

    for db_file_name in db_files:
        db_file_path = os.path.join(db_folder_path, db_file_name)
        perf_connection = sqlite3.connect(db_file_path)
        df = pd.read_sql_query("SELECT * FROM table_name", perf_connection)
        results = utils.calculate_and_display_all_metrics(df=df, filter_column='Unnamed', target_column='AllINR',
                                                          display_charts=False)
        results["file_name"] = db_file_name  # Add file name for identification
        all_metrics.append(results)
        perf_connection.close()

    results_df = pd.DataFrame(all_metrics)

    st.write("### Combined Profit Factor Metrics")
    st.dataframe(results_df.style.format(precision=2), use_container_width=True, height=1000)
