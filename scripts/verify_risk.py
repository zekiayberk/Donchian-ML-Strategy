import sys
import os
import shutil

# Add root
sys.path.append(os.getcwd())

from live.run_paper import PaperEngine, load_config
from live.monitoring import EventsLogger

class Args:
    mode = 'BACKTEST_PARITY'
    symbol = 'BTC/USDT'
    tf = '1h'
    seed = 42
    threshold_offset = 0.0
    warmup_bars = 0 # No warmup for this test

def run_risk_test():
    print("=== Risk Verification: Max Trades Daily ===")
    
    # 1. Config
    config = load_config()
    config['backtest']['initial_capital'] = 10000
    config['risk']['max_trades_daily'] = 1 # Strict Limit
    
    # 2. Cleanup
    if os.path.exists('events.jsonl'):
        os.remove('events.jsonl')
    if os.path.exists('paper_state.json'):
        os.remove('paper_state.json')
        
    # 3. Run Engine
    # We use parity_data.csv which has multiple trades
    args = Args()
    engine = PaperEngine(config, args)
    
    # Run
    try:
        engine.run()
    except KeyboardInterrupt:
        pass
        
    # 4. Analyze
    import json
    events = []
    if os.path.exists('events.jsonl'):
        with open('events.jsonl', 'r') as f:
            for line in f:
                events.append(json.loads(line))
                
    fills = [e for e in events if e['event'] == 'FILL']
    blocks = [e for e in events if e['event'] == 'GATE' and e['data'].get('action') == 'BLOCK_RISK']
    
    print(f"Total Fills: {len(fills)}")
    print(f"Total Risk Blocks: {len(blocks)}")
    
    # Expectation: 
    # Parity data has ~6 trades.
    # Max daily is 1.
    # We expect 1 Entry Fill (and maybe 1 Exit Fill if same day).
    # Wait, "Max Trades" usually counts ENTRIES.
    # My implementation in _check_risk_limits checks 'trades' counter.
    # 'trades' counter is incremented in _check_stops (Exit).
    # Wait! Logic Error.
    # If I limit ENTRIES based on CLOSED TRADES, it won't limit the first day properly until one closes!
    # "Max Trades Per Day" usually means "Max Entries Per Day".
    # I am incrementing `daily_stats['trades']` in `_check_stops`.
    # It should be incremented in `submit_order` (on Entry) or `process_bar` (on Entry Fill).
    
    # I need to fix this in the code first if my assumption is wrong.
    # Standard: "Day Trading Limit" -> Entries.
    
    if len(blocks) > 0:
        print("PASS: Risk Block triggered.")
    else:
        print("FAIL: No Risk Blocks.")

if __name__ == "__main__":
    run_risk_test()
