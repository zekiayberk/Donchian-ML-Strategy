import sys
import os

# Add root
sys.path.append(os.getcwd())

from live.paper_fill_model import PaperFillModel

def run_stress_test():
    print("=== Paper Fill Model Stress Suite ===")
    
    scenarios = [
        {'name': 'Ref (Binance Normal)', 'spread': 0.0, 'slip': 5.0, 'desc': 'No Spread, 5bps Slip'},
        {'name': 'High Volatility', 'spread': 10.0, 'slip': 20.0, 'desc': '10bps Spread, 20bps Slip'},
        {'name': 'Panic Selling', 'spread': 50.0, 'slip': 100.0, 'desc': '50bps Spread, 100bps Slip'},
        {'name': 'Zero Liquidity', 'spread': 500.0, 'slip': 500.0, 'desc': '5% Spread, 5% Slip'}
    ]
    
    market_price = 50000.0
    atr = 500.0
    
    print(f"Base Price: {market_price}")
    print(f"ATR (14): {atr}")
    
    for sc in scenarios:
        print(f"\nScenario: {sc['name']} ({sc['desc']})")
        model = PaperFillModel(spread_bps=sc['spread'], slippage_bps=sc['slip'], seed=42)
        
        # Long Fill (Ask Side)
        # Should be > Market Price
        long_fill = model.calculate_fill_price('LONG', market_price, atr)
        diff_l = long_fill - market_price
        pct_l = (diff_l / market_price) * 10000 # bps
        
        # Short Fill (Bid Side)
        # Should be < Market Price
        short_fill = model.calculate_fill_price('SHORT', market_price, atr)
        diff_s = market_price - short_fill
        pct_s = (diff_s / market_price) * 10000 # bps
        
        print(f"  LONG  Fill: {long_fill:.2f} (Delta: +{diff_l:.2f} | +{pct_l:.1f} bps)")
        print(f"  SHORT Fill: {short_fill:.2f} (Delta: -{diff_s:.2f} | -{pct_s:.1f} bps)")
        
        # Assertions
        assert long_fill > market_price, "Long fill should be above mid/market"
        assert short_fill < market_price, "Short fill should be below mid/market"
        assert long_fill > short_fill, "Ask should be higher than Bid"
        
    print("\n[PASS] All Scenarios Validated.")

if __name__ == '__main__':
    run_stress_test()
