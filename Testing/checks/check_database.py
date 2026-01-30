"""
Direct database check
"""
import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

st.set_page_config(layout="wide")
st.title("🔍 Direct Database Check")

# Connect to database
db_path = Path("storage/trading.db")
st.write(f"Database path: {db_path}")
st.write(f"Exists: {db_path.exists()}")

if db_path.exists():
    try:
        conn = sqlite3.connect(str(db_path))
        
        # List all tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(tables_query, conn)
        
        st.write("### 📋 Database Tables:")
        st.dataframe(tables)
        
        # Check each table
        for table_name in tables['name']:
            st.write(f"#### 📊 Table: `{table_name}`")
            try:
                count_query = f"SELECT COUNT(*) as count FROM {table_name};"
                count_df = pd.read_sql_query(count_query, conn)
                st.write(f"Rows: {count_df.iloc[0]['count']}")
                
                # Show sample data
                if count_df.iloc[0]['count'] > 0:
                    sample_query = f"SELECT * FROM {table_name} LIMIT 5;"
                    sample_df = pd.read_sql_query(sample_query, conn)
                    st.dataframe(sample_df)
                else:
                    st.info("No data in this table")
            except:
                st.warning(f"Could not read table {table_name}")
            
            st.write("---")
        
        conn.close()
        st.success("✅ Database check completed")
        
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        import traceback
        st.error(traceback.format_exc())
else:
    st.error("❌ Database file not found!")