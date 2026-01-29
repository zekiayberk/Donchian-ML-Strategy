
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

@dataclass
class Trade:
    """Tamamlanmış bir işlemi temsil eder."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_percent: float
    exit_reason: str # 'STOP_LOSS', 'TRAILING_STOP', 'SIGNAL', 'FORCE_CLOSE'
    commission: float
    ml_active: bool = False
    ml_prob: float = 1.0
    ml_fold: int = -1

@dataclass
class Position:
    """Açık bir pozisyonu temsil eder."""
    symbol: str
    direction: str # 'LONG' or 'SHORT'
    entry_time: datetime
    entry_price: float
    qty: float
    stop_loss: float
    initial_stop_loss: float = 0.0 # Başlangıç stop seviyesi
    take_profit: float = None # Bu stratejide kullanılmıyor ama opsiyonel
    highest_price: float = 0.0 # Trailing stop için: Long ise gördüğü en yüksek, Short ise en düşük
    lowest_price: float = float('inf') 
    ml_active: bool = False
    ml_prob: float = 1.0
    ml_fold: int = -1    
    def __post_init__(self):
        if self.initial_stop_loss == 0.0:
            self.initial_stop_loss = self.stop_loss
            
        if self.direction == 'LONG':
            self.highest_price = self.entry_price
        else:
            self.lowest_price = self.entry_price

    def update_trailing_stop(self, current_price, current_atr, k_atr):
        """
        Trailing stop günceller.
        """
        if self.direction == 'LONG':
            # Fiyat yükseldikçe stop yukarı gelir
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            # Yeni potansiyel stop
            new_stop = current_price - (k_atr * current_atr)
            if new_stop > self.stop_loss:
                self.stop_loss = new_stop
                
        elif self.direction == 'SHORT':
            # Fiyat düştükçe stop aşağı gelir
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                
            new_stop = current_price + (k_atr * current_atr)
            if new_stop < self.stop_loss:
                self.stop_loss = new_stop
