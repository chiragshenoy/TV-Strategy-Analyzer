import os
import sqlite3
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

@st.cache_data
def get_db_files(folder):
    """Get the list of database files."""
    return [f for f in os.listdir(folder) if f.endswith(".db")]

def load_data_from_db(DB_FOLDER, db_file):
    """Load data from a specific table in a database."""
    db_path = os.path.join(DB_FOLDER, db_file)
    try:
        # Create a new connection for each query
        with sqlite3.connect(db_path) as conn:
            query = f"SELECT * FROM table_name;"
            df = pd.read_sql(query, conn)
            df["db_name"] = db_file  # Add a column to identify the database
        return df
    except Exception as e:
        st.warning(f"Could not load data from {db_file}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

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