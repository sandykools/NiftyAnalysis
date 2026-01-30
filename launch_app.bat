@echo off
echo ===========================================
echo 🚀 LAUNCHING TRADING RESEARCH PLATFORM v2.0
echo ===========================================
echo.
echo Environment: trading_tool
echo Features:
echo • OI Velocity Analytics
echo • Gamma Exposure (GEX) Tracking  
echo • Structural Walls/Traps Detection
echo • Spot Divergence Analysis
echo.
echo Market Hours: 9:15 AM - 3:30 PM IST
echo.
echo Starting app...
echo URL: http://localhost:8501
echo Press Ctrl+C to stop
echo.

REM Activate conda environment
call conda activate trading_tool

REM Set Streamlit config
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

REM Run the app
streamlit run app.py

pause