#!/usr/bin/env python3
"""
Trade Analytics Demo
"""

from structured_logger import StructuredLogger

def main():
    print("TRADE ANALYTICS DEMO")
    print("=" * 40)
    
    # Create logger and analyze trades
    logger = StructuredLogger()
    analysis = logger.analyze_trades()
    
    print(f"Total Trades: {analysis.get('total_trades', 0)}")
    print(f"Win Rate: {analysis.get('win_rate', 0):.1%}")
    print(f"Total PnL: ${analysis.get('total_pnl', 0):.2f}")
    print(f"Profit Factor: {analysis.get('profit_factor', 0):.2f}")
    print(f"Avg Trade Duration: {analysis.get('avg_duration_bars', 0):.1f} bars")
    
    # Show recent trades
    trades = logger.read_trade_events()
    if not trades.empty:
        print(f"\nRecent Trade Events: {len(trades)} total")
        print("Last 3 trades:")
        for _, trade in trades.tail(3).iterrows():
            event_type = trade.get('event_type', 'N/A')
            action = trade.get('action', 'N/A')
            timestamp = str(trade.get('timestamp', 'N/A'))[:19]  # Truncate timestamp
            print(f"  {timestamp}: {event_type} - {action}")
    else:
        print("\nNo trade events found")

if __name__ == "__main__":
    main()
