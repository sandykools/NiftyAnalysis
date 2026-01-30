#!/usr/bin/env python3
"""
Fix signals table by adding missing pnl column.
"""

import sqlite3
from pathlib import Path

def fix_signals_table():
    db_path = Path("storage/trading.db")
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check if pnl column exists
    cursor.execute("PRAGMA table_info(signals)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "pnl" not in columns:
        print("Adding pnl column to signals table...")
        cursor.execute("ALTER TABLE signals ADD COLUMN pnl REAL DEFAULT 0.0")
        conn.commit()
        print("✓ Added pnl column")
    else:
        print("✓ pnl column already exists")
    
    # Also check for other missing columns
    expected_columns = ["signal_type", "status", "confidence", "pnl", "created_at"]
    for col in expected_columns:
        if col not in columns:
            print(f"Missing column: {col}")
    
    conn.close()

if __name__ == "__main__":
    fix_signals_table()