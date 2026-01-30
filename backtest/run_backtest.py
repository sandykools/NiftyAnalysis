from backtest.engine import BacktestEngine
from backtest.metrics import compute_backtest_metrics
from ml.feature_contract import FEATURE_VERSION

engine = BacktestEngine(
    feature_version=FEATURE_VERSION,
    holding_minutes=5,
    quantity=1
)

results = engine.run()
metrics = compute_backtest_metrics(results)

print("Backtest Results:")
print(results)

print("\nBacktest Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")
