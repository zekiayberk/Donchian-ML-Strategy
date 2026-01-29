
import pandas as pd
import json
import argparse
import sys
from datetime import datetime
import os

def load_events(path='events.jsonl'):
    data = []
    with open(path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return data

def generate_report(events_path, output_dir='reports'):
    events = load_events(events_path)
    if not events:
        print("No events found.")
        return

    df = pd.DataFrame(events)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter for today (UTC) or all time if requested
    today = datetime.utcnow().strftime('%Y-%m-%d')
    
    # 1. Trade Analysis (FIFO from Fills)
    fills = [e for e in events if e['event'] == 'FILL']
    trades = []
    
    # Simple FIFO Matching
    # Position tracking
    position = 0.0
    entry_pool = [] # list of (qty, price)
    
    closed_trades = []
    
    for f in fills:
        d = f['data']
        side = d['side']
        qty = float(d['qty'])
        price = float(d['price'])
        
        # Direction: LONG buys, SHORT sells
        # Normalization: Signed Qty
        signed_qty = qty if side == 'LONG' else -qty
        
        # FIFO Matching
        remaining_qty = abs(signed_qty)
        current_side = 'LONG' if signed_qty > 0 else 'SHORT'
        
        # If we have open positions and the side is opposite, we are CLOSING
        while remaining_qty > 0 and entry_pool:
            head = entry_pool[0] # FIFO
            head_side = head['side']
            
            if head_side != current_side:
                # Closing Trade
                match_qty = min(remaining_qty, head['qty'])
                
                # Calculate PnL
                # Long Entry (Buy) -> Short Exit (Sell): (Exit - Entry) * Qty
                # Short Entry (Sell) -> Long Exit (Buy): (Entry - Exit) * Qty
                if head_side == 'LONG':
                    pnl = (price - head['price']) * match_qty
                else:
                    pnl = (head['price'] - price) * match_qty
                    
                closed_trades.append({
                    'entry_price': head['price'],
                    'exit_price': price,
                    'qty': match_qty,
                    'side': head_side,
                    'pnl': pnl
                })
                
                # Update State
                head['qty'] -= match_qty
                remaining_qty -= match_qty
                
                if head['qty'] < 1e-9:
                    entry_pool.pop(0) # Fully closed this entry
            else:
                # Adding to same side
                break
                
        # If remaining qty > 0, add to pool
        if remaining_qty > 0:
            entry_pool.append({
                'qty': remaining_qty,
                'price': price,
                'side': current_side
            })

    # Summarize Closed Trades
    total_pnl = sum([t['pnl'] for t in closed_trades])
    win_count = len([t for t in closed_trades if t['pnl'] > 0])
    loss_count = len([t for t in closed_trades if t['pnl'] <= 0])
    trade_count = len(closed_trades)
    winrate = (win_count / trade_count * 100) if trade_count > 0 else 0.0 
        
    # Stats
    # Allow simple stats from daily stats if available in 'system' or 'metrics' updates?
    # Better: Use LATENCY events
    lat_evts = [e['data'] for e in events if e['event'] == 'LATENCY']
    lat_df = pd.DataFrame(lat_evts)
    
    lat_stats = "No Latency Data"
    if not lat_df.empty:
        p95 = lat_df['data_latency_ms'].quantile(0.95)
        avg = lat_df['data_latency_ms'].mean()
        lat_stats = f"p95: {p95:.2f}ms | Avg: {avg:.2f}ms"
        
    # Rejects & Mismatches
    rejects = len([e for e in events if e['event'] == 'ORDER_REJECTED'])
    mismatches = len([e for e in events if e['event'] == 'RECONCILE_MISMATCH'])
    
    # Fills Count
    fill_count = len(fills)
    
    report = f"""# Daily Report: {today}

## Execution Health
- **Latency**: {lat_stats}
- **Rejects**: {rejects}
- **Reconcile Mismatches**: {mismatches}

## Trading Activity
- **Fills**: {fill_count}
- **Trades Closed**: {trade_count}
- **Win Rate**: {winrate:.1f}% ({win_count}W / {loss_count}L)
- **Realized PnL**: ${total_pnl:.2f}

## Alerts
"""
    # Append Alerts
    alerts = [e for e in events if e['event'] in ['KILL_SWITCH_TRIGGERED', 'LIVE_ARM_FAILED', 'RECONCILE_MISMATCH']]
    if alerts:
        for a in alerts:
            report += f"- **{a['event']}**: {a['data']}\n"
    else:
        report += "- Clean Run (No Critical Alerts)\n"
        
    # Write to File
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    outfile = os.path.join(output_dir, f"daily_{today}.md")
    with open(outfile, 'w') as f:
        f.write(report)
        
    print(f"Report generated: {outfile}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', default='events.jsonl')
    args = parser.parse_args()
    
    generate_report(args.events)
