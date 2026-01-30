from datetime import datetime, time, timedelta, timezone
from typing import Callable, Optional, Dict
import threading
import time as time_module  # Rename to avoid conflict with datetime.time
import streamlit as st
import numpy as np  # Add numpy import for metrics

class MarketScheduler:
    """
    Enhanced scheduler with research-aware execution control.
    
    Features:
    - Market hours detection
    - Intelligent interval adjustment
    - Execution throttling
    - Performance monitoring
    - Error recovery
    """
    
    def __init__(
        self,
        interval_seconds: int = 30,
        market_open: time = time(9, 15),  # CORRECT: use time objects
        market_close: time = time(15, 30),  # CORRECT: use time objects
        pre_market_minutes: int = 15,
        post_market_minutes: int = 15
    ):
        self.interval_seconds = interval_seconds
        self.market_open = market_open
        self.market_close = market_close
        self.pre_market_minutes = pre_market_minutes
        self.post_market_minutes = post_market_minutes
        
        # Execution tracking
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._consecutive_failures: int = 0
        self._total_executions: int = 0
        self._lock = threading.Lock()
        
        # Performance metrics
        self.execution_times = []
        self.error_log = []
        
        # Adaptive interval (can adjust based on market conditions)
        self.min_interval = 10  # seconds
        self.max_interval = 300  # seconds
        self._current_interval = interval_seconds
        
    # ==============================
    # TIME MANAGEMENT
    # ==============================
    
    def _is_market_open(self, now: datetime) -> bool:
        """
        Check if market is open, including pre/post market.
        """
        t = now.time()
        
        # Pre-market period
        pre_market_start = time(
            self.market_open.hour,
            max(0, self.market_open.minute - self.pre_market_minutes)  # Ensure minutes don't go negative
        )
        
        # Post-market period
        post_market_end_minute = self.market_close.minute + self.post_market_minutes
        post_market_end_hour = self.market_close.hour + (post_market_end_minute // 60)
        post_market_end_minute = post_market_end_minute % 60
        post_market_end = time(
            post_market_end_hour,
            post_market_end_minute
        )
        
        return pre_market_start <= t <= post_market_end
    
    def _is_core_market_hours(self, now: datetime) -> bool:
        """Check if it's core trading hours."""
        t = now.time()
        return self.market_open <= t <= self.market_close
    
    def _get_time_to_market_open(self, now: datetime) -> Optional[float]:
        """Get seconds until market opens."""
        market_open_today = datetime.combine(now.date(), self.market_open)
        
        if now < market_open_today:
            return (market_open_today - now).total_seconds()
        
        # Market already open today, check tomorrow
        tomorrow = now.date() + timedelta(days=1)
        market_open_tomorrow = datetime.combine(tomorrow, self.market_open)
        return (market_open_tomorrow - now).total_seconds()
    
    def _get_market_status(self, now: datetime) -> Dict[str, any]:
        """Get detailed market status."""
        status = {
            "is_market_open": self._is_market_open(now),
            "is_core_hours": self._is_core_market_hours(now),
            "current_time": now,
            "market_open": self.market_open,
            "market_close": self.market_close
        }
        
        if not status["is_market_open"]:
            time_to_open = self._get_time_to_market_open(now)
            if time_to_open:
                status["time_to_open_hours"] = time_to_open / 3600
                status["next_open"] = now + timedelta(seconds=time_to_open)
        
        return status
    
    # ==============================
    # EXECUTION MANAGEMENT
    # ==============================
    
    def _is_due(self, now: datetime) -> bool:
        """Check if execution is due."""
        if self._last_run is None:
            return True
        
        delta = (now - self._last_run).total_seconds()
        return delta >= self._current_interval
    
    def _adjust_interval(self, execution_time: float, success: bool):
        """
        Adaptively adjust execution interval based on performance.
        """
        if not success:
            self._consecutive_failures += 1
            
            # Increase interval on consecutive failures
            if self._consecutive_failures >= 3:
                self._current_interval = min(
                    self._current_interval * 1.5,
                    self.max_interval
                )
                self._consecutive_failures = 0
        else:
            self._consecutive_failures = 0
            self._last_success = datetime.now()
            
            # Adjust interval based on execution time
            if execution_time > self._current_interval * 0.8:
                # Execution taking too long, increase interval
                self._current_interval = min(
                    self._current_interval * 1.2,
                    self.max_interval
                )
            elif execution_time < self._current_interval * 0.3:
                # Execution fast, can decrease interval (but not below min)
                self._current_interval = max(
                    self._current_interval * 0.9,
                    self.min_interval
                )
    
    def _should_execute(self, now: datetime, market_regime: Optional[str] = None) -> bool:
        """
        Determine if execution should proceed based on multiple factors.
        """
        # Basic checks
        if not self._is_market_open(now):
            return False
        
        if not self._is_due(now):
            return False
        
        # Check for too many recent failures
        if self._consecutive_failures >= 5:
            if self._last_success:
                time_since_success = (now - self._last_success).total_seconds()
                if time_since_success < 300:  # 5 minutes
                    return False  # Wait after multiple failures
        
        # Market regime-based adjustments
        if market_regime:
            # Execute more frequently during volatile regimes
            if market_regime in ["SQUEEZE", "BREAKOUT", "ACCELERATING"]:
                self._current_interval = max(self.min_interval, self.interval_seconds * 0.7)
            # Execute less frequently during stable regimes
            elif market_regime in ["RANGING", "STABILIZING"]:
                self._current_interval = min(self.max_interval, self.interval_seconds * 1.3)
        
        return True
    
    # ==============================
    # EXECUTION ENTRYPOINT
    # ==============================
    
    def run_if_due(self, run_cycle: Callable[[], any], market_regime: Optional[str] = None) -> Dict[str, any]:
        """
        Execute run_cycle() if due, with enhanced monitoring.
        
        Returns:
            Execution result dictionary
        """
        now = datetime.now()
        
        # Check if should execute
        if not self._should_execute(now, market_regime):
            return {
                "executed": False,
                "reason": "Not due or market closed",
                "next_execution_in": self._get_next_execution_time(now),
                "market_status": self._get_market_status(now)
            }
        
        # Thread-safe execution
        with self._lock:
            # Double-check inside lock
            if not self._is_due(now):
                return {
                    "executed": False,
                    "reason": "Already executed by another thread",
                    "next_execution_in": self._get_next_execution_time(now)
                }
            
            execution_start = datetime.now()
            result = {
                "executed": True,
                "start_time": execution_start.isoformat(),
                "success": False,
                "error": None,
                "execution_time": 0.0
            }
            
            try:
                # Execute the cycle
                cycle_result = run_cycle()
                execution_end = datetime.now()
                execution_time = (execution_end - execution_start).total_seconds()
                
                # Update tracking
                self._last_run = execution_start
                self._total_executions += 1
                self.execution_times.append(execution_time)
                
                # Keep only recent execution times
                if len(self.execution_times) > 100:
                    self.execution_times = self.execution_times[-100:]
                
                # Update result
                result.update({
                    "success": True,
                    "execution_time": execution_time,
                    "end_time": execution_end.isoformat(),
                    "cycle_result": cycle_result,
                    "current_interval": self._current_interval
                })
                
                # Adjust interval based on performance
                self._adjust_interval(execution_time, success=True)
                
            except Exception as e:
                execution_end = datetime.now()
                execution_time = (execution_end - execution_start).total_seconds()
                
                # Log error
                self.error_log.append({
                    "timestamp": execution_start.isoformat(),
                    "error": str(e),
                    "execution_time": execution_time
                })
                
                # Keep error log manageable
                if len(self.error_log) > 50:
                    self.error_log = self.error_log[-50:]
                
                # Update result
                result.update({
                    "success": False,
                    "error": str(e),
                    "execution_time": execution_time,
                    "end_time": execution_end.isoformat()
                })
                
                # Adjust interval on failure
                self._adjust_interval(execution_time, success=False)
                
                # Update last run time even on failure (to prevent rapid retry)
                self._last_run = execution_start
                self._total_executions += 1
            
            return result
    
    # ==============================
    # MONITORING & METRICS
    # ==============================
    
    def get_metrics(self) -> Dict[str, any]:
        """Get scheduler performance metrics."""
        metrics = {
            "total_executions": self._total_executions,
            "current_interval": self._current_interval,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_success": self._last_success.isoformat() if self._last_success else None,
            "consecutive_failures": self._consecutive_failures,
            "recent_error_count": len(self.error_log[-10:]),
            "is_market_open_now": self._is_market_open(datetime.now())
        }
        
        # Add execution time statistics
        if self.execution_times:
            metrics.update({
                "avg_execution_time": np.mean(self.execution_times),
                "max_execution_time": np.max(self.execution_times),
                "min_execution_time": np.min(self.execution_times),
                "recent_avg_execution_time": np.mean(self.execution_times[-10:]) if len(self.execution_times) >= 10 else None
            })
        
        return metrics
    
    def _get_next_execution_time(self, now: datetime) -> float:
        """Get seconds until next execution."""
        if self._last_run is None:
            return 0.0
        
        next_run = self._last_run + timedelta(seconds=self._current_interval)
        return max((next_run - now).total_seconds(), 0.0)
    
    def reset(self):
        """Reset scheduler state."""
        with self._lock:
            self._last_run = None
            self._last_success = None
            self._consecutive_failures = 0
            self._total_executions = 0
            self.execution_times = []
            self.error_log = []
            self._current_interval = self.interval_seconds
    
    def set_interval(self, interval_seconds: int):
        """Dynamically set execution interval."""
        with self._lock:
            self.interval_seconds = interval_seconds
            self._current_interval = max(
                min(interval_seconds, self.max_interval),
                self.min_interval
            )

# ==============================
# UTILITY FUNCTIONS
# ==============================

def create_market_scheduler(
    interval_seconds: int = 30,
    market_open: str = "09:15",
    market_close: str = "15:30"
) -> MarketScheduler:
    """
    Factory function to create market scheduler.
    
    Args:
        interval_seconds: Execution interval in seconds
        market_open: Market open time (HH:MM)
        market_close: Market close time (HH:MM)
    """
    # Parse time strings
    open_hour, open_minute = map(int, market_open.split(":"))
    close_hour, close_minute = map(int, market_close.split(":"))
    
    return MarketScheduler(
        interval_seconds=interval_seconds,
        market_open=time(open_hour, open_minute),  # CORRECT
        market_close=time(close_hour, close_minute)  # CORRECT
    )

def display_scheduler_status(scheduler: MarketScheduler):
    """Display scheduler status in Streamlit."""
    if not scheduler:
        return
    
    metrics = scheduler.get_metrics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if metrics["is_market_open_now"]:
            st.success("🟢 Market Open")
        else:
            st.warning("🔴 Market Closed")
        
        next_exec = scheduler._get_next_execution_time(datetime.now())
        if next_exec > 0:
            st.metric("Next Execution", f"{int(next_exec)}s")
    
    with col2:
        st.metric("Interval", f"{metrics['current_interval']}s")
        
        if metrics.get("avg_execution_time"):
            st.metric("Avg Exec Time", f"{metrics['avg_execution_time']:.1f}s")
    
    with col3:
        st.metric("Total Executions", metrics["total_executions"])
        
        if metrics["recent_error_count"] > 0:
            st.error(f"Recent Errors: {metrics['recent_error_count']}")