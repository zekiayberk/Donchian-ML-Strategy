
import json
import os
from datetime import datetime

class StateManager:
    """
    Botun durumunu (açık pozisyonlar, trade geçmişi, son çalışma zamanı)
    yerel bir dosyada saklar.
    """
    def __init__(self, filename='bot_state.json'):
        self.filename = filename
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"State yüklenemedi: {e}. Yeni state oluşturuluyor.")
        
        return {
            'last_update': None,
            'current_position': None, # {symbol, direction, entry_price, qty, stop_loss}
            'total_pnl': 0.0,
            'trades': []
        }

    def save_state(self):
        self.state['last_update'] = datetime.now().isoformat()
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            print(f"State kaydedilemedi: {e}")

    def update_position(self, position_data):
        """
        :param position_data: Dict or None (if closed)
        """
        self.state['current_position'] = position_data
        self.save_state()

    def add_trade(self, trade_info):
        self.state['trades'].append(trade_info)
        # PnL update logic if needed
        self.save_state()
