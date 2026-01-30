import pandas as pd
import numpy as np


def compute_backtest_metrics(trades_df: pd.DataFrame) -> dict:
    """
    Compute performance metrics from backtest trades.
    Safe for empty DataFrames.
    """

    if trades_df is None or trades_df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "expectancy": 0.0
        }

    pnl = trades_df["pnl"]

    total_trades = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    win_rate = len(wins) / total_trades if total_trades else 0.0
    total_pnl = pnl.sum()
    avg_pnl = pnl.mean()

    # Equity curve
    equity = pnl.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = drawdown.min()

    # Sharpe (PnL-based)
    sharpe = 0.0
    if pnl.std() != 0:
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(len(pnl))

    # Expectancy
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = losses.mean() if not losses.empty else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 3),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe": round(sharpe, 3),
        "expectancy": round(expectancy, 2)
    }
