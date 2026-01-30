import os
import shutil
from pathlib import Path

print("=" * 60)
print("FORCE CLEANUP - ENSURING LOGIN BUTTON APPEARS")
print("=" * 60)

# Remove all token files and directories
token_paths = [
    "data/tokens",
    "tokens",
    ".streamlit/tokens",
    "upstox_tokens.enc",
    "data/upstox_tokens.enc"
]

for path in token_paths:
    if os.path.exists(path):
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
                print(f"✓ Removed directory: {path}")
            except Exception as e:
                print(f"✗ Error removing {path}: {e}")
        else:
            try:
                os.remove(path)
                print(f"✓ Removed file: {path}")
            except Exception as e:
                print(f"✗ Error removing {path}: {e}")

# Clear .env file
env_file = ".env"
if os.path.exists(env_file):
    try:
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Remove UPSTOX tokens but keep other variables
        new_lines = []
        for line in lines:
            if not line.startswith("UPSTOX_"):
                new_lines.append(line)
        
        with open(env_file, 'w') as f:
            f.writelines(new_lines)
        
        print(f"✓ Cleared UPSTOX tokens from {env_file}")
    except Exception as e:
        print(f"✗ Error clearing .env: {e}")

# Create fresh token directory
os.makedirs("data/tokens", exist_ok=True)
print("✓ Created fresh token directory")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. Run the fix below to update session.py")
print("2. Then run: streamlit run app.py")
print("3. You should now see the login button")