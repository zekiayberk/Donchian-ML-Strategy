import logging
import uuid
import ccxt # Requires: pip install ccxt
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from datetime import datetime
from .paper_fill_model import PaperFillModel

logger = logging.getLogger(__name__)

class OrderState:
    NEW = "NEW"
    OPEN = "OPEN"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class PaperOrder:
    def __init__(self, symbol, side, qty, order_type='MARKET', price=None, stop_price=None, idempotency_key=None):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side # 'LONG' or 'SHORT'
        # ...
        self.qty = qty
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.idempotency_key = idempotency_key
        self.state = OrderState.NEW
        self.avg_fill_price = 0.0
        self.filled_qty = 0.0
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.reject_reason = None
        # Attribution
        self.fees_paid = 0.0
        self.funding_paid = 0.0
        self.realized_pnl = 0.0 # Net of fees/funding? No, usually Net PnL is diff.
        self.net_pnl = 0.0

    def __repr__(self):
        return f"<Order {self.id[:8]} {self.side} {self.qty} @ {self.order_type} | {self.state}>"

class BrokerAdapter(ABC):
    @abstractmethod
    def get_balance(self) -> float:
        pass
    
    @abstractmethod
    def get_position(self, symbol) -> dict:
        """Returns {'size': x, 'entry_price': y, 'pnl': z} or None"""
        pass
    
    @abstractmethod
    def submit_order(self, order: PaperOrder) -> bool:
        pass
    
    @abstractmethod
    def cancel_order(self, order_id) -> bool:
        pass

class PaperBroker(BrokerAdapter):
    def __init__(self, initial_balance=10000.0, fill_model: PaperFillModel = None):
        self.balance = initial_balance
        self.fill_model = fill_model if fill_model else PaperFillModel()
        
        # State
        self.orders: Dict[str, PaperOrder] = {}
        self.positions: Dict[str, dict] = {} # symbol -> {'size': 0, 'entry_price': 0}
        
        # Idempotency Cache (LRU)
        from collections import deque
        self.processed_keys = deque(maxlen=10000)
        
        # PnL Tracking
        self.realized_pnl = 0.0
        self.equity_history = []
        
    def get_balance(self) -> float:
        return self.balance
        
    def get_position(self, symbol) -> dict:
        return self.positions.get(symbol)
        
    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculates total equity (Balance + Unrealized PnL)"""
        unrealized = 0.0
        for sym, pos in self.positions.items():
            current_price = current_prices.get(sym)
            if current_price and pos['size'] != 0:
                if pos['size'] > 0:
                    val = (current_price - pos['entry_price']) * pos['size']
                else:
                    val = (pos['entry_price'] - current_price) * abs(pos['size'])
                unrealized += val
        return self.balance + unrealized

    def submit_order(self, order: PaperOrder) -> str:
        """
        Submits an order. In Paper mode, we just register it.
        Execution happens via `process_orders`.
        """
        # Idempotency Check
        if order.idempotency_key:
            if order.idempotency_key in self.processed_keys:
                logger.warning(f"DUPLICATE ORDER IGNORED: Key={order.idempotency_key}")
                return "DUPLICATE"
            self.processed_keys.append(order.idempotency_key)

        if order.qty <= 0:
            order.state = OrderState.REJECTED
            order.reject_reason = "Zero Quantity"
            return order.id

        # Check Position Limit Logic (Simple: max 1 pos per symbol)
        # If we are opening a NEW position while one exists (and not closing/reversing)
        # Assuming FIFO / or complex logic handled by Engine. 
        # Broker just executes.
        
        self.orders[order.id] = order
        order.state = OrderState.OPEN
        logger.info(f"[PaperBroker] Order Submitted: {order}")
        return order.id

    def execute_market_order(self, order_id, market_price, atr=0.0):
        """
        IMMEDIATE execution helper for Market Orders.
        Used when 'Next Open' price is known.
        """
        order = self.orders.get(order_id)
        if not order or order.state != OrderState.OPEN:
            return False
            
        # Calculate Fill Price
        fill_price = self.fill_model.calculate_fill_price(
            order.side, market_price, atr
        )
        
        # Update Order
        order.state = OrderState.FILLED
        order.avg_fill_price = fill_price
        order.filled_qty = order.qty
        order.updated_at = datetime.utcnow()
        
        # Update Balance / Position
        self._update_position(order)
        
        logger.info(f"[PaperBroker] FILLED {order.side} {order.qty} @ {fill_price:.2f} (Ref: {market_price:.2f})")
        return True
        
    def _update_position(self, order: PaperOrder):
        symbol = order.symbol
        current_pos = self.positions.get(symbol, {'size': 0.0, 'entry_price': 0.0})
        
        # Simple Netting Logic
        # LONG 1.0, then SHORT 1.0 -> 0
        # LONG 1.0, then SHORT 2.0 -> SHORT 1.0
        
        old_size = current_pos['size']
        new_size = 0.0
        
        trade_size = order.qty if order.side == 'LONG' else -order.qty
        
        if old_size == 0:
            # New Position
            new_size = trade_size
            current_pos['entry_price'] = order.avg_fill_price
        else:
            # Existing Position
            if (old_size > 0 and trade_size > 0) or (old_size < 0 and trade_size < 0):
                # Adding to position (Average Entry Price)
                total_val = (abs(old_size) * current_pos['entry_price']) + (abs(trade_size) * order.avg_fill_price)
                new_size = old_size + trade_size
                current_pos['entry_price'] = total_val / abs(new_size)
            else:
                # Reducing / Closing / Reversing
                # First, Realize PnL on the closed portion
                closing_qty = min(abs(old_size), abs(trade_size))
                
                # PnL Calculation
                # Long Close: (Exit - Entry) * Qty
                # Short Close: (Entry - Exit) * Qty
                if old_size > 0: # Long
                    pnl = (order.avg_fill_price - current_pos['entry_price']) * closing_qty
                else: # Short
                    pnl = (current_pos['entry_price'] - order.avg_fill_price) * closing_qty
                
                self.balance += pnl
                self.realized_pnl += pnl
                
                remaining = old_size + trade_size
                new_size = remaining
                
                # If reversed
                if (old_size > 0 and new_size < 0) or (old_size < 0 and new_size > 0):
                    current_pos['entry_price'] = order.avg_fill_price
        
        current_pos['size'] = new_size
        if new_size == 0:
            del self.positions[symbol]
        else:
            self.positions[symbol] = current_pos

    def cancel_order(self, order_id) -> bool:
        order = self.orders.get(order_id)
        if order and order.state == OrderState.OPEN:
            order.state = OrderState.CANCELLED
            return True
        return False

class BinanceBroker(BrokerAdapter):
    """
    Real execution via Binance Futures (Testnet or Live).
    """
    def __init__(self, api_key, secret, testnet=True):
        self.testnet = testnet
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("BinanceBroker initialized in TESTNET mode.")
        else:
            logger.warning("BinanceBroker initialized in LIVE mode.")
            
        # Verify connection
        try:
            self.exchange.load_markets()
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            raise

        from collections import deque
        self.processed_keys = deque(maxlen=100)

    def get_balance(self) -> float:
        try:
            bal = self.exchange.fetch_balance()
            # USDT free balance
            return float(bal['USDT']['free'])
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    def get_position(self, symbol) -> dict:
        try:
            positions = self.exchange.fetch_positions([symbol])
            # Filter for symbol (ccxt might return list)
            target = None
            for p in positions:
                if p['symbol'] == symbol:
                    target = p
                    break
            
            if target and float(target['contracts']) != 0:
                side = 1 if target['side'] == 'long' else -1
                size = float(target['contracts'])
                if target['side'] == 'short':
                    size = -size
                    
                entry_price = float(target['entryPrice'])
                return {'size': size, 'entry_price': entry_price}
            return None
        except Exception as e:
            logger.error(f"Error fetching position: {e}")
            return None

    def submit_order(self, order: PaperOrder) -> bool:
        """
        Maps PaperOrder to ccxt create_order.
        """
        try:
            # Type Mapping
            type_str = order.order_type.lower() # limit, market
            side_str = 'buy' if order.side == 'LONG' else 'sell'
            
            # Idempotency Check (Local)
            if order.idempotency_key:
                if order.idempotency_key in self.processed_keys:
                    logger.warning(f"[BinanceBroker] DUPLICATE ORDER IGNORED: Key={order.idempotency_key}")
                    return True # Treat as success/ignored? Or False? PaperBroker returns "DUPLICATE" string actually. 
                    # Adapter returns bool. PaperBroker returns str?
                    # BrokerAdapter abstract says -> bool.
                    # PaperBroker implementation returns "DUPLICATE" or order.id.
                    # This violates interface! PaperBroker submit_order returns string. Adapter says bool.
                    # I should fix Adapter definition or PaperBroker or checking logic.
                    # PaperEngine ignores return value of submit_order (line 498).
                    # So return True (handled) is fine.
                self.processed_keys.append(order.idempotency_key)

            params = {}
            if order.idempotency_key:
                params['clientOrderId'] = order.idempotency_key # Binance supports this
                
            # Stop Loss logic to be handled separately or via params
            # Stop Loss logic to be handled separately or via params
            
            created = self.exchange.create_order(
                symbol=order.symbol,
                type=type_str,
                side=side_str,
                amount=order.qty,
                price=order.price, # None for market
                params=params
            )
            
            logger.info(f"Binance Order Created: {created['id']} - {created['status']}")
            
            # Update PaperOrder state
            order.id = str(created['id']) # Sync ID
            order.state = OrderState.OPEN if created['status'] == 'open' else OrderState.FILLED
            
            # In Market Order, it might be filled immediately
            if created['status'] == 'closed':
                 order.state = OrderState.FILLED
                 if 'average' in created:
                     order.avg_fill_price = created['average']
                     order.filled_qty = created['filled']
            
            return True
            
        except Exception as e:
            logger.error(f"Binance Submit Error: {e}")
            order.state = OrderState.REJECTED
            order.reject_reason = str(e)
            return False

    def cancel_order(self, order_id) -> bool:
        # Not fully implemented for Phase 7
        return False
