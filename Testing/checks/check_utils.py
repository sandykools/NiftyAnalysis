"""
Check if utils folder and files exist
"""
import os
from pathlib import Path

print("🔍 Checking utils folder structure...")

# Check project structure
project_root = Path(".")
utils_folder = project_root / "utils"
expiry_utils = utils_folder / "expiry_utils.py"

print(f"Project root: {project_root.absolute()}")
print(f"Utils folder: {utils_folder.absolute()}")
print(f"Utils exists: {utils_folder.exists()}")
print(f"expiry_utils.py: {expiry_utils.absolute()}")
print(f"expiry_utils.py exists: {expiry_utils.exists()}")

# List files in utils if it exists
if utils_folder.exists():
    print("\n📁 Files in utils folder:")
    for file in utils_folder.iterdir():
        print(f"  - {file.name}")

# Check if we can import it
print("\n🔧 Testing import...")
try:
    import sys
    sys.path.append(str(project_root))
    
    from utils.expiry_utils import get_trading_expiry, is_market_open
    print("✅ Successfully imported from utils.expiry_utils")
    
    # Test the functions
    expiry = get_trading_expiry("NIFTY")
    print(f"  get_trading_expiry('NIFTY') = {expiry}")
    
    market_status = is_market_open()
    print(f"  is_market_open() = {market_status}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    
    # Create the file if missing
    print("\n📝 Creating missing expiry_utils.py...")
    expiry_content = '''"""
Utility functions for expiry date selection
"""
from datetime import datetime, time
from data.instrument_master import get_available_expiries

def is_market_open() -> bool:
    """
    Check if market is currently open.
    NSE market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
    """
    now = datetime.now()
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday (5) or Sunday (6)
        return False
    
    # Check time
    market_open = time(9, 15)  # 9:15 AM IST
    market_close = time(15, 30)  # 3:30 PM IST
    
    current_time = now.time()
    return market_open <= current_time <= market_close

def get_trading_expiry(underlying: str) -> str:
    """
    Get appropriate expiry date for trading.
    - If market is open and today is expiry: use today
    - If market is closed and today is expiry: use next expiry
    - Otherwise: use nearest expiry
    """
    expiries = get_available_expiries(underlying)
    if not expiries:
        return ""
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Check if today is an expiry day
    if today in expiries:
        if is_market_open():
            # Market is open on expiry day - use today
            return today
        else:
            # Market closed on expiry day - use next expiry
            for expiry in sorted(expiries):
                if expiry > today:
                    return expiry
            return expiries[-1]  # Fallback to last expiry
    else:
        # Today is not expiry day - get nearest future expiry
        for expiry in sorted(expiries):
            if expiry >= today:
                return expiry
        return expiries[-1]  # Fallback to last expiry
'''
    
    # Create utils folder if it doesn't exist
    utils_folder.mkdir(exist_ok=True)
    
    # Write the file
    with open(expiry_utils, 'w') as f:
        f.write(expiry_content)
    
    print(f"✅ Created {expiry_utils}")
    
    # Try import again
    try:
        from utils.expiry_utils import get_trading_expiry, is_market_open
        print("✅ Now successfully imported!")
    except Exception as e2:
        print(f"❌ Still can't import: {e2}")

except Exception as e:
    print(f"❌ Other error: {e}")