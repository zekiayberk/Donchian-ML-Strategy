import random
import numpy as np

class PaperFillModel:
    """
    Simulates execution fills (Price & Quantity) for Paper Trading.
    Supports deterministic mode for parity checks.
    """
    def __init__(self, spread_bps=0.0, slippage_bps=0.0, seed=None):
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random.Random()
        
    def calculate_fill_price(self, side, market_price, atr=0.0, use_random=False):
        """
        Calculates the filled price for a MARKET order.
        
        :param side: 'LONG' or 'SHORT' (Direction of the trade)
        :param market_price: The triggering price (e.g., Open of the bar)
        :param atr: Average True Range (volatility proxy)
        :param use_random: If True, adds random noise relative to ATR
        :return: float (Filled Price)
        """
        
        # Base Cost (Spread + Slippage)
        # Assuming market_price is 'Mid', so we pay half spread + full slippage?
        # User Rule: fill = Ask * (1+slip)
        # Let's approximate: Fill = Price * (1 + (Spread/2 + Slippage)/10000)
        
        cost_bps = self.spread_bps / 2 + self.slippage_bps
        cost_factor = cost_bps / 10000.0
        
        # Random Slippage Component (Optional - for advanced realism)
        # If use_random is True, we add noise propotional to ATR
        random_impact = 0.0
        if use_random and atr > 0:
            # E.g. +/- 5% of ATR as extra noise
            noise = self.rng.uniform(-0.05, 0.05) 
            random_impact = atr * noise
        
        # Calculate Final Price
        if side == 'LONG':
            # Buying: Higher price is bad
            fill_price = market_price * (1 + cost_factor) + abs(random_impact)
        else: # SHORT
            # Selling: Lower price is bad
            fill_price = market_price * (1 - cost_factor) - abs(random_impact)
            
        return fill_price

    def get_slippage_stats(self):
        """Returns stats about simulated slippage (for monitoring)"""
        return {
            'spread_bps': self.spread_bps,
            'fixed_slippage_bps': self.slippage_bps,
            'deterministic': self.seed is not None
        }
