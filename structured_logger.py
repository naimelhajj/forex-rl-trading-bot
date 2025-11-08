"""
Structured Logging Module
Provides structured JSON-lines logging for trades, episodes, and events.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd


class StructuredLogger:
    """
    JSON-lines logger for structured events (trades, episodes, errors).
    """
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate loggers for different event types
        self.trade_logger = self._setup_logger('trades', 'trade_events.jsonl')
        self.episode_logger = self._setup_logger('episodes', 'episode_events.jsonl')
        self.error_logger = self._setup_logger('errors', 'error_events.jsonl')
    
    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        """Setup a JSON-lines logger for specific event type."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler for JSON lines
        handler = logging.FileHandler(self.log_dir / filename)
        handler.setLevel(logging.INFO)
        
        # No formatter needed - we'll write JSON directly
        logger.addHandler(handler)
        logger.propagate = False
        
        return logger
    
    def log_trade_open(self, timestamp: datetime, action: str, price: float, 
                      lots: float, sl_price: float, tp_price: float,
                      equity_before: float, margin_used: float, **kwargs):
        """Log trade opening event."""
        event = {
            'event_type': 'trade_open',
            'timestamp': timestamp.isoformat(),
            'action': str(action) if action is not None else "UNKNOWN",
            'price': float(price) if price is not None else 0.0,
            'lots': float(lots) if lots is not None else 0.0,
            'sl_price': float(sl_price) if sl_price is not None else None,
            'tp_price': float(tp_price) if tp_price is not None else None,
            'equity_before': float(equity_before) if equity_before is not None else 0.0,
            'margin_used': float(margin_used) if margin_used is not None else 0.0,
            **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool)) and v is not None}
        }
        self.trade_logger.info(json.dumps(event))
    
    def log_trade_close(self, timestamp: datetime, reason: str, exit_price: float,
                       pnl: float, equity_after: float, duration_bars: int, **kwargs):
        """Log trade closing event."""
        event = {
            'event_type': 'trade_close',
            'timestamp': timestamp.isoformat(),
            'reason': str(reason) if reason is not None else "UNKNOWN",
            'exit_price': float(exit_price) if exit_price is not None else 0.0,
            'pnl': float(pnl) if pnl is not None else 0.0,
            'equity_after': float(equity_after) if equity_after is not None else 0.0,
            'duration_bars': int(duration_bars) if duration_bars is not None else 0,
            **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool)) and v is not None}
        }
        self.trade_logger.info(json.dumps(event))
    
    def log_trade_sl_move(self, timestamp: datetime, old_sl: float, new_sl: float,
                         current_price: float, **kwargs):
        """Log stop-loss movement event."""
        event = {
            'event_type': 'sl_move',
            'timestamp': timestamp.isoformat(),
            'old_sl': float(old_sl),
            'new_sl': float(new_sl),
            'current_price': float(current_price),
            **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}
        }
        self.trade_logger.info(json.dumps(event))
    
    def log_episode_start(self, episode: int, timestamp: datetime, **kwargs):
        """Log episode start event."""
        event = {
            'event_type': 'episode_start',
            'episode': int(episode),
            'timestamp': timestamp.isoformat(),
            **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}
        }
        self.episode_logger.info(json.dumps(event))
    
    def log_episode_end(self, episode: int, timestamp: datetime, reward: float,
                       final_equity: float, steps: int, trades: int, **kwargs):
        """Log episode end event."""
        event = {
            'event_type': 'episode_end',
            'episode': int(episode),
            'timestamp': timestamp.isoformat(),
            'reward': float(reward),
            'final_equity': float(final_equity),
            'steps': int(steps),
            'trades': int(trades),
            **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}
        }
        self.episode_logger.info(json.dumps(event))
    
    def log_validation(self, episode: int, timestamp: datetime, fitness: float,
                      sharpe: float, cagr: float, **kwargs):
        """Log validation event."""
        event = {
            'event_type': 'validation',
            'episode': int(episode),
            'timestamp': timestamp.isoformat(),
            'fitness': float(fitness),
            'sharpe': float(sharpe),
            'cagr': float(cagr),
            **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}
        }
        self.episode_logger.info(json.dumps(event))
    
    def log_error(self, timestamp: datetime, error_type: str, message: str,
                 context: Optional[Dict[str, Any]] = None):
        """Log error event."""
        event = {
            'event_type': 'error',
            'timestamp': timestamp.isoformat(),
            'error_type': error_type,
            'message': str(message),
            'context': context or {}
        }
        self.error_logger.info(json.dumps(event))
    
    def read_trade_events(self, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Read trade events as DataFrame for analysis."""
        trade_file = self.log_dir / 'trade_events.jsonl'
        if not trade_file.exists():
            return pd.DataFrame()
        
        events = []
        with open(trade_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    event_time = pd.to_datetime(event['timestamp'])
                    
                    # Filter by time range if specified
                    if start_time and event_time < start_time:
                        continue
                    if end_time and event_time > end_time:
                        continue
                    
                    events.append(event)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        if not events:
            return pd.DataFrame()
        
        df = pd.DataFrame(events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    def read_episode_events(self) -> pd.DataFrame:
        """Read episode events as DataFrame for analysis."""
        episode_file = self.log_dir / 'episode_events.jsonl'
        if not episode_file.exists():
            return pd.DataFrame()
        
        events = []
        with open(episode_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        
        if not events:
            return pd.DataFrame()
        
        df = pd.DataFrame(events)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        return df
    
    def analyze_trades(self) -> Dict[str, Any]:
        """Analyze logged trades and return summary statistics."""
        trades_df = self.read_trade_events()
        if trades_df.empty:
            return {'error': 'No trade data available'}
        
        # Separate open and close events
        opens = trades_df[trades_df['event_type'] == 'trade_open']
        closes = trades_df[trades_df['event_type'] == 'trade_close']
        
        if opens.empty or closes.empty:
            return {'error': 'Incomplete trade data'}
        
        # Basic statistics
        total_trades = len(closes)
        profitable_trades = len(closes[closes['pnl'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = closes['pnl'].sum()
        avg_pnl = closes['pnl'].mean()
        
        profit_factor = (closes[closes['pnl'] > 0]['pnl'].sum() / 
                        abs(closes[closes['pnl'] < 0]['pnl'].sum())) if closes[closes['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
        
        avg_duration = closes['duration_bars'].mean()
        
        # Trade reasons analysis
        close_reasons = closes['reason'].value_counts().to_dict()
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'profit_factor': profit_factor,
            'avg_duration_bars': avg_duration,
            'close_reasons': close_reasons,
            'best_trade': float(closes['pnl'].max()) if not closes.empty else 0,
            'worst_trade': float(closes['pnl'].min()) if not closes.empty else 0,
        }


if __name__ == "__main__":
    # Test the structured logger
    logger = StructuredLogger()
    
    # Simulate some trade events
    now = datetime.now()
    logger.log_trade_open(now, 'LONG', 1.1000, 0.1, 1.0950, 1.1100, 1000.0, 20.0)
    logger.log_trade_close(now, 'TP_HIT', 1.1100, 10.0, 1010.0, 50)
    
    # Analyze trades
    analysis = logger.analyze_trades()
    print("Trade Analysis:")
    for k, v in analysis.items():
        print(f"  {k}: {v}")
