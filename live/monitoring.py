
import json
import logging
import time
from datetime import datetime
from collections import deque
import atexit
import sys

# Try standard timezone, fallback to backports
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

class EventsLogger:
    """
    Logs structured events to a JSONL file.
    Events: SIGNAL, GATE, ORDER, FILL, ERROR, STATUS
    """
    def __init__(self, filename='events.jsonl'):
        self.filename = filename
        self.file = open(filename, 'a')
        # Ensure file is closed on exit
        atexit.register(self.close)
        
    def log_event(self, event_type, data):
        """
        :param event_type: str (e.g. 'SIGNAL')
        :param data: dict
        """
        payload = {
            'timestamp': datetime.now(ZoneInfo("UTC")).isoformat(),
            'event': event_type,
            'data': data
        }
        json_line = json.dumps(payload)
        self.file.write(json_line + '\n')
        self.file.flush()
        
        self._check_alert(event_type, data)

    def _check_alert(self, event_type, data):
        ALERT_EVENTS = {
            'KILL_SWITCH_TRIGGERED', 
            'RECONCILE_MISMATCH', 
            'ORDER_REJECTED', 
            'LATENCY_WARN',
            'LIVE_ARM_FAILED'
        }
        
        if event_type in ALERT_EVENTS:
            # RED BANNER
            msg = f"\n\033[41m\033[1m !!! CRITICAL ALERT: {event_type} !!! \033[0m\n"
            msg += f"Data: {data}\n"
            sys.stderr.write(msg)
            sys.stderr.flush()

    def close(self):
        if self.file:
            self.file.close()

class ConsoleDashboard:
    """
    Prints a refreshed dashboard to the console.
    """
    def __init__(self):
        self.last_events = deque(maxlen=5)
        
    def add_event(self, event_type, msg):
        ts = datetime.now(ZoneInfo("UTC")).strftime('%H:%M:%S')
        self.last_events.append(f"[{ts}] {event_type}: {msg}")
        
    def refresh(self, equity, balance, positions, active_orders, metrics):
        """
        Clears screen (ansi) and repaints.
        """
        # ANSI Clear Screen
        print("\033[H\033[J", end="")
        
        print(f"=== PAPER TRADING DASHBOARD ===")
        print(f"Time: {datetime.now(ZoneInfo('UTC')).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Equity: ${equity:.2f} | Balance: ${balance:.2f}")
        print("-" * 30)
        
        print("POSITIONS:")
        if not positions:
            print("  [No Active Positions]")
        else:
            for sym, pos in positions.items():
                size = pos['size']
                entry = pos['entry_price']
                print(f"  {sym}: {size} @ {entry:.2f}")

        print("-" * 30)
        print("METRICS:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
            
        print("-" * 30)
        print("RECENT EVENTS:")
        for evt in self.last_events:
            print(f"  {evt}")
            
class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.errors = 0
        self.signals = 0
        self.orders = 0
        self.fills = 0
        
    @property
    def uptime(self):
        return int(time.time() - self.start_time)
        
    def to_dict(self):
        return {
            'Uptime (s)': self.uptime,
            'Errors': self.errors,
            'Signals': self.signals,
            'Orders': self.orders,
            'Fills': self.fills
        }
    
    def from_dict(self, data):
        if not data: return
        self.errors = data.get('Errors', 0)
        self.signals = data.get('Signals', 0)
        self.orders = data.get('Orders', 0)
        self.fills = data.get('Fills', 0)
