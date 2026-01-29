
import ccxt
import os
import json

class BinanceFuturesBroker:
    """
    Canlı işlem adaptörü (Binance Futures).
    CCXT kütüphanesini kullanır.
    """
    def __init__(self, api_key, api_secret, testnet=True):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {
                'defaultType': 'future',
            }
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
            print("Broker: TESTNET modunda başlatıldı.")
            
        self.exchange.load_markets()
        
    def get_balance(self):
        """Kullanılabilir USDT bakiyesi"""
        balance = self.exchange.fetch_balance()
        return balance['USDT']['free']
        
    def get_position(self, symbol):
        """Mevcut pozisyonu getir"""
        # Symbol format: BTC/USDT
        positions = self.exchange.fetch_positions([symbol])
        for p in positions:
            if p['symbol'] == symbol:
                return {
                    'qty': float(p['contracts']),
                    'entryPrice': float(p['entryPrice']),
                    'side': p['side'].upper() if p['side'] else None,
                    'unrealizedPnl': float(p['unrealizedPnl'])
                }
        return None

    def create_order(self, symbol, side, qty, price=None, params={}):
        """
        Market veya Limit emir gönderir.
        :param side: 'buy' or 'sell'
        """
        type = 'limit' if price else 'market'
        print(f"EMİR GÖNDERİLİYOR: {symbol} {side} {qty} @ {price if price else 'MARKET'}")
        
        try:
            order = self.exchange.create_order(symbol, type, side, qty, price, params)
            print(f"EMİR BAŞARILI: {order['id']}")
            return order
        except Exception as e:
            print(f"EMİR HATASI: {e}")
            return None
            
    def cancel_all_orders(self, symbol):
        self.exchange.cancel_all_orders(symbol)

class PaperBroker(BinanceFuturesBroker):
    """
    Sadece log basan, gerçek işlem yapmayan broker.
    """
    def __init__(self):
        print("Broker: PAPER TRADING modunda başlatıldı (Sanal).")
        self.balance = 10000
        self.positions = {}
        
    def get_balance(self):
        return self.balance
        
    def get_position(self, symbol):
        return self.positions.get(symbol, {'qty': 0, 'side': None})
        
    def create_order(self, symbol, side, qty, price=None, params={}):
        print(f"[PAPER] EMİR: {symbol} {side} {qty} type={price}")
        # Basit state update simülasyonu yapılabilir ama
        # canlı bot genelde borsadan position çeker.
        # Paper modda internal state tutmak gerekir.
        if side == 'buy':
            self.positions[symbol] = {'qty': qty, 'entryPrice': price or 100, 'side': 'LONG'}
        elif side == 'sell':
            self.positions[symbol] = {'qty': 0, 'entryPrice': 0, 'side': None} # Close varsayımı
        return {'id': 'paper_123'}

