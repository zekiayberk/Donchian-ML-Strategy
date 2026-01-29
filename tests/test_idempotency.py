import unittest
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from live.execution import PaperBroker, PaperOrder

class TestIdempotency(unittest.TestCase):
    def test_duplicate_submission(self):
        broker = PaperBroker()
        
        # Order 1
        key1 = "BTC/USDT|1h|2024-01-01 10:00|LONG"
        order1 = PaperOrder('BTC/USDT', 'LONG', 1.0, idempotency_key=key1)
        
        res1 = broker.submit_order(order1)
        print(f"Order 1 Result: {res1}")
        self.assertNotEqual(res1, "DUPLICATE")
        
        # Order 1 Duplicate (Same Key, different object)
        order1_dup = PaperOrder('BTC/USDT', 'LONG', 1.0, idempotency_key=key1)
        res1_dup = broker.submit_order(order1_dup)
        print(f"Order 1 Dup Result: {res1_dup}")
        self.assertEqual(res1_dup, "DUPLICATE")
        
        # Order 2 (New Key)
        key2 = "BTC/USDT|1h|2024-01-01 11:00|LONG"
        order2 = PaperOrder('BTC/USDT', 'LONG', 1.0, idempotency_key=key2)
        res2 = broker.submit_order(order2)
        print(f"Order 2 Result: {res2}")
        self.assertNotEqual(res2, "DUPLICATE")

if __name__ == '__main__':
    unittest.main()
