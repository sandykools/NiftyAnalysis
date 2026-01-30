"""
Database Initialization Script.
Run this once to set up the database with research features.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from storage.repository import initialize_database, migrate_database
from ml.feature_contract import FEATURE_VERSION

def main():
    """Initialize and migrate database."""
    print("=" * 60)
    print("DATABASE INITIALIZATION SCRIPT")
    print("=" * 60)
    
    try:
        # Initialize database
        print("\n1. Initializing database...")
        initialize_database()
        
        # Migrate to current feature version
        print(f"\n2. Migrating to feature version {FEATURE_VERSION}...")
        success = migrate_database(FEATURE_VERSION)
        
        if success:
            print(f"\n✅ Database initialized successfully!")
            print(f"   Feature Version: {FEATURE_VERSION}")
            print(f"   Database Path: storage/trading.db")
        else:
            print("\n❌ Database migration failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()