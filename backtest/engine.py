import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from core.signals.state_machine import SignalStateMachine


DB_PATH = Path("G:/trading_app/storage/trading.db")


class BacktestEngine:
    """
    Replays historical features to simulate signals and trades.
    """

    def __init__(
        self,
        feature_version: str,
        holding_minutes: int = 5,
        quantity: int = 1
    ):
        self.feature_version = feature_version
        self.holding_minutes = holding_minutes
        self.quantity = quantity
        self.signal_engine = SignalStateMachine(signal_expiry_minutes=holding_minutes)

    # ==============================
    # LOAD HISTORICAL FEATURES
    # ==============================

    def load_features(self) -> pd.DataFrame:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(
                """
                SELECT *
                FROM market_features
                WHERE feature_version = ?
                ORDER BY timestamp
                """,
                conn,
                params=(self.feature_version,)
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ==============================
    # RUN BACKTEST
    # ==============================

    def run(self) -> pd.DataFrame:
        df = self.load_features()
        if df.empty:
            raise RuntimeError("No features found for backtest")

        trades = []
        open_trade = None

        for i in range(len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]

            # If no open trade → check for signal
            if open_trade is None:
                signal = self.signal_engine.build_signal(row)

                if signal["signal_type"] in ("BUY", "SELL"):
                    open_trade = {
                        "signal_id": signal["signal_id"],
                        "direction": signal["signal_type"],
                        "entry_time": next_row["timestamp"],
                        "entry_price": next_row["spot_price"],
                        "expiry_time": next_row["timestamp"]
                                       + timedelta(minutes=self.holding_minutes)
                    }

            # If trade open → check exit
            if open_trade is not None:
                if next_row["timestamp"] >= open_trade["expiry_time"]:
                    exit_price = next_row["spot_price"]

                    pnl = (
                        (exit_price - open_trade["entry_price"])
                        if open_trade["direction"] == "BUY"
                        else (open_trade["entry_price"] - exit_price)
                    ) * self.quantity

                    trades.append({
                        "signal_id": open_trade["signal_id"],
                        "entry_time": open_trade["entry_time"],
                        "exit_time": next_row["timestamp"],
                        "entry_price": open_trade["entry_price"],
                        "exit_price": exit_price,
                        "quantity": self.quantity,
                        "pnl": pnl
                    })

                    open_trade = None

        return pd.DataFrame(trades)
