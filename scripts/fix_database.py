"""
Database Fix Script - Run this to fix migration issues.
"""

import sys
from pathlib import Path
import sqlite3

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


# Add this function to scripts/fix_database.py or create a new script

def add_details_json_column():
    """Add missing details_json column to system_health table."""
    try:
        with get_connection() as conn:
            # Check if column exists
            cur = conn.execute("PRAGMA table_info(system_health)")
            columns = [row[1] for row in cur.fetchall()]
            
            if "details_json" not in columns:
                print("Adding details_json column to system_health table...")
                conn.execute("""
                    ALTER TABLE system_health 
                    ADD COLUMN details_json TEXT DEFAULT '{}'
                """)
                print("✓ Added details_json column")
            else:
                print("✓ details_json column already exists")
    except Exception as e:
        print(f"❌ Error adding details_json column: {e}")
        
def fix_database():
    """Fix database migration issues."""
    print("=" * 60)
    print("DATABASE FIX SCRIPT")
    print("=" * 60)
    
    db_path = Path("G:/trading_app/storage/trading.db")
    
    if not db_path.exists():
        print("Database file not found. Creating new database...")
        from storage.repository import initialize_storage
        initialize_storage()
        return
    
    print(f"Fixing database at: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(str(db_path))
    
    try:
        # Check if market_features table exists
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_features'")
        if not cur.fetchone():
            print("market_features table doesn't exist. Creating...")
            conn.close()
            from storage.repository import initialize_storage
            initialize_storage()
            return
        
        # Get current columns
        cur = conn.execute("PRAGMA table_info(market_features)")
        columns = [row[1] for row in cur.fetchall()]
        print(f"Current columns in market_features: {columns}")
        
        # Research columns that should exist
        research_columns = [
            "oi_velocity", "oi_velocity_ma", "oi_velocity_std",
            "oi_regime_expansive", "oi_regime_constricted",
            "net_gamma", "gamma_regime_positive", "gamma_regime_negative",
            "gamma_flip_distance", "max_gamma_strike_distance",
            "wall_strength", "wall_defense_score", "trap_probability",
            "price_oi_divergence", "price_gamma_divergence", "divergence_score",
            "has_divergence", "max_pain_distance", "vix_smile", "skewness",
            "spring_detection", "upthrust_detection", "accumulation_score",
            "gamma_wall_interaction", "velocity_divergence_composite", "trap_gamma_composite"
        ]
        
        # Add missing columns
        for column in research_columns:
            if column not in columns:
                print(f"Adding column: {column}")
                try:
                    conn.execute(f"ALTER TABLE market_features ADD COLUMN {column} REAL DEFAULT 0.0")
                    print(f"  ✓ Added {column}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e):
                        print(f"  ⚠️ Column {column} already exists")
                    else:
                        print(f"  ❌ Error adding {column}: {e}")
        
        # Check database_info table
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='database_info'")
        if not cur.fetchone():
            print("Creating database_info table...")
            conn.execute("""
                CREATE TABLE database_info (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
        
        # Set feature version to v2.0
        from datetime import datetime
        conn.execute(
            "INSERT OR REPLACE INTO database_info (key, value, updated_at) VALUES (?, ?, ?)",
            ("feature_version", "v2.0", datetime.utcnow().isoformat())
        )
        
        conn.commit()
        print("✓ Database fixed successfully!")
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    fix_database()