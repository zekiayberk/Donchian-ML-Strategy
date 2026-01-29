
import unittest
from unittest.mock import MagicMock, patch
import logging
import sys
import os

# Add local path to import modules
sys.path.append(os.getcwd())

from live.run_paper import PaperEngine
from live.execution import BinanceBroker, PaperOrder

class TestSafety(unittest.TestCase):
    def setUp(self):
        # Silence logging
        logging.getLogger('PaperEngine').setLevel(logging.CRITICAL)
        
        # Mock Config and Args
        self.config = {
            'data': {'symbol': 'BTC/USDT', 'timeframe': '1h'},
            'backtest': {'slippage_bps': 1, 'commission_bps': 4, 'initial_capital': 10000},
            'ml': {'enabled': False, 'threshold': 0.7},
            'strategy': {'donchian_period': 24, 'atr_period': 14}
        }
        self.args = MagicMock()
        self.args.symbol = 'BTC/USDT'
        self.args.tf = '1h'
        self.args.mode = 'LIVE'
        self.args.feed = 'LIVE'
        self.args.live_confirm = 'mock_hash'
        self.args.seed = 42 # Fix seed type error
        
    @patch('live.execution.ccxt')
    def test_reconciliation_mismatch(self, mock_ccxt):
        """
        Simulate Broker returning 1.0 position but Local State saying 0.0.
        Expect panic (HALTED_OPS).
        """
        # Mock Engine components
        engine = PaperEngine(self.config, self.args)
        
        # Mock Broker (BinanceBroker)
        # Note: PaperEngine logic instantiates Broker in __init__. 
        # But we want to mock the instance it uses.
        engine.broker = MagicMock()
        engine.broker.get_position.return_value = {'size': 1.0} # Remote has 1.0
        
        # Mock StateStore
        engine.state_store = MagicMock()
        engine.state_store.load.return_value = {'position': {'size': 0.0}} # Local has 0.0
        
        # Mock Logger IO
        engine.logger_io = MagicMock()
        
        # Force Mode
        engine.mode = 'LIVE'
        
        # Run Reconcile
        try:
            engine._reconcile_broker_state()
        except RuntimeError:
            pass # We changed logic to Soft Halt (just logging), but verify engine.state
            
        # Verify
        self.assertEqual(engine.state, 'HALTED_OPS', "Engine should trigger Soft Halt on mismatch")
        engine.logger_io.log_event.assert_called_with('RECONCILE_MISMATCH', {'local': 0.0, 'remote': 1.0, 'diff': 1.0})
        print("\n[Passed] Reconciliation Mismatch triggered HALT.")

    @patch('live.execution.ccxt')
    def test_idempotency_binance(self, mock_ccxt):
        """
        Test BinanceBroker local idempotency cache.
        """
        # Setup Mock Exchange
        mock_exchange = MagicMock()
        mock_exchange.create_order.return_value = {'id': '12345', 'status': 'open'}
        mock_ccxt.binance.return_value = mock_exchange
        
        # Init Broker
        broker = BinanceBroker('key', 'secret', testnet=True)
        
        # Create Order with Key
        order = PaperOrder('BTC/USDT', 'LONG', 1.0, idempotency_key='bar_1_long')
        
        # 1st Submission
        res = broker.submit_order(order)
        self.assertTrue(res)
        mock_exchange.create_order.assert_called_once()
        
        # 2nd Submission (Duplicate)
        res2 = broker.submit_order(order)
        self.assertTrue(res2) # Should return True (handled)
        
        # Verify create_order was NOT called again
        self.assertEqual(mock_exchange.create_order.call_count, 1, "Duplicate order should use cache, not API call")
        print("\n[Passed] Idempotency blocked duplicate API call.")

if __name__ == '__main__':
    unittest.main()
