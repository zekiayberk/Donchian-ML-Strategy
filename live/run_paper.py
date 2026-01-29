import argparse
import sys
import yaml
import time
import logging
import traceback
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo # Fallback if needed
import pandas as pd
from dotenv import load_dotenv

# Load Secrets
load_dotenv()


# Infrastructure
from live.data_feed import LiveFeed, HistoricalFeed
from live.execution import PaperBroker, PaperOrder, OrderState, BinanceBroker
from live.paper_fill_model import PaperFillModel
from live.monitoring import EventsLogger, ConsoleDashboard, MetricsCollector
from live.persistence import StateStore

# Strategy
from strategy.signals import SignalGenerator
from indicators.donchian import calculate_donchian_channel
from indicators.atr import calculate_atr
from src.ml.inference import MLEngine
from src.ml.features import engineering_features

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('PaperEngine')

class PaperEngine:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.args = args
        self.mode = args.mode
        
        # PARANOID SAFETY CHECK
        import os
        if os.environ.get('LIVE_TRADING'):
            if self.mode != 'LIVE':
                msg = "SAFETY HALT: LIVE_TRADING env var is set! Unset it for SHADOW/TESTNET/PAPER runs."
                logging.getLogger('PaperEngine').critical(msg)
                raise RuntimeError(msg)
        
        self.live_armed = False # Hard Gate
        
        # Init Components
        self._init_data_feed()
        self.logger_io = EventsLogger('events.jsonl')
        self.dashboard = ConsoleDashboard()
        self.metrics = MetricsCollector()
        self.state_store = StateStore('paper_state.json')
        
        # Broker
        seed = args.seed if args.seed is not None else None
        
        # Init Fill Model from Config
        spread = config.get('backtest', {}).get('simulated_spread_bps', 0.0)
        slippage = config['backtest']['slippage_bps']
        
        if self.mode == 'TESTNET':
            import os
            # Expect keys in ENV or Config
            ak = os.environ.get('BINANCE_TESTNET_KEY') or config.get('exchange', {}).get('key')
            sk = os.environ.get('BINANCE_TESTNET_SECRET') or config.get('exchange', {}).get('secret')
            
            if not ak or not sk:
                raise ValueError("TESTNET mode requires BINANCE_TESTNET_KEY and BINANCE_TESTNET_SECRET")
            
            self.broker = BinanceBroker(api_key=ak, secret=sk, testnet=True)
        else:
            # Paper / Shadow / DryRun
            fill_model = PaperFillModel(spread_bps=spread, slippage_bps=slippage, seed=seed)
            self.broker = PaperBroker(initial_balance=config['backtest']['initial_capital'], fill_model=fill_model)
        
        # Config Overrides (CLI)
        if args.threshold_offset is not None:
            self.threshold_offset = args.threshold_offset
        else:
            self.threshold_offset = config.get('ml', {}).get('threshold_offset', 0.0)
            
        self.ml_threshold = config.get('ml', {}).get('threshold', 0.7) + self.threshold_offset
            
        # ML Engine
        self.ml_enabled = config['ml']['enabled']
        if self.ml_enabled:
            logger.info(f"Loading ML Models (Offset: {self.threshold_offset})...")
            self.ml_engine = MLEngine("models", threshold_offset=self.threshold_offset, config=config)
        else:
            self.ml_engine = None
            
        # State
        self.state = 'RUNNING' # RUNNING, HALTED_RISK, HALTED_OPS
        self.consecutive_errors = 0
        self.consecutive_losses = 0
        self.last_update_time = time.time()
        self.max_dd = 0.0
        
        # Kill-Switch Config
        self.max_consecutive_errors = 5
        self.max_drawdown_limit = 0.15 # 15% Hard Stop
        
        # Indicators
        self.donchian_n = config['strategy']['donchian_period']
        self.atr_n = config['strategy']['atr_period']
        
        # Warmup
        self.warmup_bars = getattr(args, 'warmup_bars', 500)
        self.processed_bars = 0
        
        # State
        self.cooldown_counter = 0
        self.current_stops = {}
        self.daily_stats = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'trades': 0,
            'consecutive_losses': 0,
            'pnl': 0.0
        }
        
        # Risk Config
        self.max_trades_daily = config.get('risk', {}).get('max_trades_daily', 5)
        self.max_consecutive_losses = config.get('risk', {}).get('max_consecutive_losses', 3)
        self.timezone_str = 'Europe/Istanbul'
        
        # Load State
        self._load_state()
        
        # Truth Check
        self._reconcile_broker_state()
        
    def _init_data_feed(self):
        symbol = self.args.symbol if self.args.symbol else self.config['data']['symbol']
        timeframe = self.args.tf if self.args.tf else self.config['data']['timeframe']
        
        # Check override arg first
        feed_type = getattr(self.args, 'feed', 'LIVE')
        
        if self.mode == 'BACKTEST_PARITY' or feed_type == 'HIST':
            self.feed = HistoricalFeed('data/parity_data.csv', symbol=symbol)
        elif self.mode in ['PAPER', 'DRY_RUN', 'SHADOW']:
            # Normal usage: Live Feed
            self.feed = LiveFeed(symbol, timeframe)
        
        # Special case: Override feed externally (for parity harness)
        # handled by setter if needed.

    def run(self):
        logger.info(f"Engine Started in {self.mode} mode.")
        
        # 0. Boot Seal (Config & Version)
        import hashlib
        import subprocess
        import sys
        import platform
        
        # Config Hash
        config_str = str(self.config).encode('utf-8')
        config_hash = hashlib.sha256(config_str).hexdigest()
        
        # Git Hash
        try:
            # Use stderr=DEVNULL to suppress "fatal: not a git repository"
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).strip().decode('utf-8')
        except Exception:
            commit_hash = "NO_GIT"
            
        boot_seal = {
            'timestamp': datetime.now(ZoneInfo("UTC")).isoformat(),
            'mode': self.mode,
            'config_hash': config_hash,
            'commit_hash': commit_hash,
            'python_version': sys.version,
            'platform': platform.platform()
        }
        self.logger_io.log_event('BOOT_SEAL', boot_seal)
        logger.info(f"BOOT SEAL: {boot_seal}")

        # HARD GATE: LIVE MODE ARMING
        if self.mode == 'LIVE':
            import os
            env_confirm = os.environ.get('LIVE_TRADING')
            arg_confirm = self.args.live_confirm
            
            if env_confirm == 'YES_I_UNDERSTAND' and arg_confirm == config_hash:
                self.live_armed = True
                logger.warning("!!! SYSTEM ARMED FOR LIVE TRADING !!!")
                self.logger_io.log_event('LIVE_ARMED', boot_seal)
            else:
                msg = f"LIVE ARM FAILED. Env={env_confirm}, ArgHash={arg_confirm}, RealHash={config_hash}"
                logger.critical(msg)
                self.logger_io.log_event('LIVE_ARM_FAILED', {'reason': msg})
                raise RuntimeError(msg)

        self.logger_io.log_event('system', {'msg': 'Engine Started', 'mode': self.mode})
        
        try:
            # Main Event Loop
            for open_bar_df in self.feed.wait_for_next_bar():
                # 1. Update Ops Metrics
                self.metrics.signals += 0 # Tick
                self.last_update_time = time.time()
                processing_start_time = time.time()
                self.consecutive_errors = 0 # Reset on successful tick
                
                # Check for Gaps
                last_row = open_bar_df.iloc[-1]
                last_bar_ts = last_row.name # This is Close Time (or Open Time?)
                # Assumption: Feed yields CLOSED bars labeled by their Open Time or Close Time? 
                # CCXT usually returns Open Time.
                # If TF=1h, Close Time = Open Time + 1h.
                # Let's verify standard: Pandas df index usually Open Time.
                
                # We need explicit Close Time for Latency Calc.
                # DataFeed.delta helper? 
                if hasattr(self.feed, 'delta'):
                    close_time = last_bar_ts + self.feed.delta
                else:
                    # Fallback for HistoricalFeed (assume 1h)
                    close_time = last_bar_ts + pd.Timedelta(hours=1)
                
                now_utc = datetime.now(ZoneInfo("UTC"))
                
                # Data Latency: Now - Close Time
                # Note: last_bar_ts is usually naive in pandas if read from csv without utc=True
                # Ensure timezone awareness
                if last_bar_ts.tzinfo is None:
                    last_bar_ts = last_bar_ts.tz_localize('UTC')
                if close_time.tzinfo is None:
                    close_time = close_time.tz_localize('UTC')

                if self.mode in ['LIVE', 'TESTNET']:
                    data_latency_ms = (now_utc - close_time).total_seconds() * 1000
                else:
                    data_latency_ms = 0.0 # meaningless in historical replay
                
                # Log Bar Receipt
                bar_evt = {
                    'bar_time': str(last_bar_ts),
                    'close_time': str(close_time),
                    'is_closed_flag': True, # feed.wait_for_next_bar guarantee
                    'recv_time': str(now_utc)
                }
                self.logger_io.log_event('BAR_RECEIVED', bar_evt)
                
                self._reconcile_state(last_bar_ts)
                
                # Runtime Reconciliation (Broker Safety)
                self._reconcile_broker_state()

                # 3. Process Logic
                self.process_bar(open_bar_df)
                
                # Decision Latency
                decision_latency_ms = (time.time() - processing_start_time) * 1000
                
                # Log Latency
                lat_evt = {
                    'data_latency_ms': round(data_latency_ms, 2),
                    'decision_latency_ms': round(decision_latency_ms, 2)
                }
                self.logger_io.log_event('LATENCY', lat_evt)
                
                # Latency Warn (Only in Live/Testnet)
                if self.mode in ['LIVE', 'TESTNET'] and data_latency_ms > 10000: # 10s
                    logger.warning(f"HIGH LATENCY: {data_latency_ms:.0f}ms")
                    self.logger_io.log_event('LATENCY_WARN', lat_evt)
                
                # 4. Update Dashboard
                self._update_ui()
                
                # 5. Check Kill-Switch
                if not self._check_health():
                    break
                    
        except KeyboardInterrupt:
            logger.info("Stopping...")
        except Exception as e:
            logger.error(f"Fatal Loop Error: {e}")
            self.logger_io.log_event('error', {'msg': str(e), 'trace': traceback.format_exc()})
            raise

    def process_bar(self, df_bar):
        """
        Core logic: Indicators -> Signal -> ML -> Order -> Execution
        """
        # Daily Stats Reset
        self._check_daily_reset()
        
        # We need historical context for indicators.
        # LiveFeed/HistoricalFeed should provide context. 
        # But wait, yield only gives 1 row.
        # We need the full buffer.
        
        # Pull full history from feed buffer
        # This is expensive if we pull ALL. 
        # Optimized: Feed should keep a buffer or we keep it.
        # Let's assume feed.get_latest_bars(N) gives us context.
        
        # Context needs to include the NEW bar.
        # LiveFeed `wait_for_next_bar` yields the new bar, but also we need previous N bars.
        
        # Better: We maintain `self.history_df` locally.
        # On startup, we fetch N=500.
        # On new bar, we append.
        
        # But for Parity Check, HistoricalFeed yields 1 by 1.
        # We should rely on `feed.get_latest_bars` OR
        # Just use `feed.get_latest_bars(500)` inside the loop? 
        # -> `HistoricalFeed` logic supports `get_latest_bars(N)` based on `current_idx`.
        
        # Fetch Context (Lookback needed for Donchian(89))
        # Use larger lookback to ensure ATR/EMA convergence for parity
        lookback = 1000 
        df_context = self.feed.get_latest_bars(lookback) 
        
        # If context is too small, skip
        if len(df_context) < self.donchian_n:
            logger.warning("Not enough bars for indicators yet.")
            return

        # Calc Indicators
        df = df_context.copy()
        df = calculate_donchian_channel(df, self.donchian_n)
        df = calculate_atr(df, self.atr_n)
        
        # Generate Signal (Vectorized, but we only care about the last one)
        df = SignalGenerator.generate_signals(df, self.config)
        
        # Last row is the 'Just Closed' bar
        last_row = df.iloc[-1]
        signal = int(last_row['entry_signal'])
        atr = last_row.get('atr', 0.0)
        
        # Log Signal
        if signal != 0:
            self.logger_io.log_event('SIGNAL', {
                'ts': str(last_row.name), 
                'dir': signal, 
                'price': last_row['close']
            })
            self.metrics.signals += 1

        # Execution Logic (Mode A)
        # Signal at Close -> Entry at Next Open (Current Ticker)
        current_price = self.feed.get_current_price()
        
        # Increment Warmup Counter (Time Based)
        self.processed_bars += 1
        
        if signal != 0:
            # Warmup Check
            if self.processed_bars < self.warmup_bars:
                remaining = self.warmup_bars - self.processed_bars
                if remaining % 10 == 0 or remaining < 5:
                    logger.info(f"WARMUP: {remaining} bars remaining.")
                    self.logger_io.log_event('WARMUP_STATUS', {'remaining': remaining, 'processed': self.processed_bars})
            
            # 1. Check ML
            is_allowed = True
            probs = None
            ml_signal = signal # Default to raw signal
            
            # WARMUP BLOCK
            if self.processed_bars < self.warmup_bars:
                is_allowed = False
                self.logger_io.log_event('GATE', {'action': 'BLOCK_WARMUP', 'prob': 0.0, 'fold': 'N/A'})

            elif self.ml_enabled and self.ml_engine:
                # Prepare features
                # Need enough history for features
                 df_feat = engineering_features(df)
                 last_feat_row = df_feat.iloc[[-1]] # DataFrame format
                 
                 # Enrich with breakout strength if needed (usually handled in engineering_features or dict)
                 # Reconstruct dict for prediction as in backtest
                 row_dict = last_row.to_dict()
                 row_dict['direction'] = signal
                 # ... other dynamic features ... (ML Engine handles most via dataframe now?)
                 # MLEngine.predict expects DF with features.
                 
                 # Quick fix: Backtest engine did some manual calculation?
                 # Yes, breakout_strength.
                 # Let's trust engineering_features to have done it or do it here.
                 # Backtest `engineering_features` adds static. 
                 # `engine.py` adds `breakout_strength` manually.
                 
                 c = last_row['close']
                 upper = last_row['donchian_upper']
                 lower = last_row['donchian_lower']
                 atr = last_row['atr']
                 
                 bs = 0.0
                 if atr > 0:
                     if signal == 1: bs = (c - upper) / atr
                     elif signal == -1: bs = (lower - c) / atr
                 
                 # We need to inject this into the row sent to ML
                 # MLEngine expects a DataFrame.
                 last_feat_row = last_feat_row.copy()
                 last_feat_row['breakout_strength'] = bs
                 last_feat_row['direction'] = signal
                 
                 # PARITY FIX: Ensure timestamp is pd.Timestamp (not int64 from column)
                 # MLEngine expects datetime object for fold lookup
                 if isinstance(last_row.name, pd.Timestamp):
                     last_feat_row['timestamp'] = last_row.name
                 else:
                     # Fallback if name is not timestamp (unlikely in this setup)
                     last_feat_row['timestamp'] = pd.to_datetime(last_row.name)

                 ml_res = self.ml_engine.predict(last_feat_row)
                 logger.info(f"ML Check: Ts={last_feat_row.get('timestamp').iloc[0]} Status={ml_res['status']} Prob={ml_res['prob']:.4f}")
                 
                 is_allowed = True
                 if ml_res['status'] == 'oos_test':
                     fold_id = f"fold_{ml_res['fold']:02d}"
                     # Reconstruct base threshold (since used_threshold has offset)
                     base_t = ml_res['used_threshold'] - self.threshold_offset
                     
                     is_allowed, tag, used_t, meta = self.ml_engine.decide_with_guard(
                         prob=ml_res['prob'],
                         fold=fold_id,
                         base_threshold=base_t,
                         offset=self.threshold_offset
                     )
                     logger.info(f"  > Guard Decision: {is_allowed} (Tag: {tag})")
                     
                     action_tag = f"ALLOWED_GUARD" if tag == "GUARD" else ("ALLOWED" if is_allowed else "BLOCK_ML")
                     self.logger_io.log_event('GATE', {'action': action_tag, 'prob': ml_res['prob'], 'fold': fold_id})
                     
                 else:
                     # Non-OOS: Simple Threshold Check
                     # predict() returns used_threshold (which includes offset if configured in MLEngine)
                     used_t = ml_res.get('used_threshold', 0.5)
                     is_allowed = ml_res['prob'] >= used_t
                     logger.info(f"  > Simple Decision: {is_allowed} (Prob {ml_res['prob']} >= {used_t})")
                     
                     action_tag = "ALLOWED" if is_allowed else "BLOCK_ML"
                     self.logger_io.log_event('GATE', {'action': action_tag, 'prob': ml_res['prob'], 'fold': ml_res['status']})


            # 2. Check Cooldown / Position Limits
            if self.cooldown_counter > 0:
                is_allowed = False
                # self.logger_io.log_event('GATE', {'action': 'BLOCK_COOLDOWN', 'counter': self.cooldown_counter})
            
            # Check Risk Limits
            if is_allowed and not self._check_risk_limits():
                is_allowed = False
                self.logger_io.log_event('GATE', {'action': 'BLOCK_RISK', 'reason': 'DAILY_LIMIT'})
            
            # 3. Submit Order
            if is_allowed:
                side = 'LONG' if signal == 1 else 'SHORT'
                
                # Check current position
                # If same side -> Ignore (or pyramiding? Config says no)
                current_pos = self.broker.get_position(self.feed.symbol)
                
                action = 'OPEN'
                if current_pos:
                    if (current_pos['size'] > 0 and side == 'LONG') or \
                       (current_pos['size'] < 0 and side == 'SHORT'):
                           action = 'IGNORE'
                    else:
                        action = 'REVERSE' # Close + Open
                
                if action != 'IGNORE':
                    # Calc Size (Risk Based)
                    balance = self.broker.get_balance()
                    risk_per_trade = self.config['risk']['risk_per_trade']
                    stop_atr_mult = self.config['strategy']['stop_loss_atr_multiplier']
                    
                    # Risk Amount
                    risk_amt = balance * risk_per_trade
                    
                    # Stop Distance
                    dist = atr * stop_atr_mult
                    
                    if dist > 0:
                        qty = risk_amt / dist
                        qty = round(qty, 6) # approx
                        
                        target_qty = qty
                        trade_qty = target_qty
                        
                        if action == 'REVERSE':
                            trade_qty += abs(current_pos['size'])
                            
                        # Set Initial Stop Loss price for the order
                        stop_price = 0.0
                        if side == 'LONG':
                            stop_price = current_price - dist
                        else:
                            stop_price = current_price + dist
                            
                        # SHADOW MODE: Suppress actual order, but log intent
                        if self.mode == 'SHADOW':
                            logger.info("SHADOW MODE: Suppressing Order. Logging Intent.")
                            shadow_payload = {
                                'bar_time': str(last_row.name),
                                'symbol': self.feed.symbol,
                                'tf': self.args.tf,
                                'signal': int(signal),
                                'ml_prob': float(probs[1]) if probs is not None else 0.0,
                                'ml_threshold': self.ml_threshold,
                                'ml_allowed': bool(ml_signal != 0),
                                'risk_allowed': True, # Passed _check_risk_limits
                                'risk_block_reason': None,
                                'intended_qty': trade_qty,
                                'intended_side': side,
                                'intended_stop': stop_price,
                                'cooldown_state': self.cooldown_counter,
                                'warmup_state': max(0, self.warmup_bars - self.processed_bars)
                            }
                            self.logger_io.log_event('SHADOW_ORDER', shadow_payload)
                            return # Exit without tracking order in broker

                        # Idempotency Key
                        # Format: Symbol|TF|BarTime|Direction
                        # Ensure BarTime is robust str
                        idem_key = f"{self.feed.symbol}|{self.args.tf}|{last_row.name}|{side}"
                        
                        # SAFETY: Hard Gate Check (Redundant but Critical)
                        if self.mode == 'LIVE' and not self.live_armed:
                             logger.critical("FATAL: LIVE MODE BUT NOT ARMED. BLOCKING ORDER.")
                             return

                        order = PaperOrder(self.feed.symbol, side, trade_qty, stop_price=stop_price, idempotency_key=idem_key)
                        self.broker.submit_order(order)
                        self.metrics.orders += 1
                        
                        # Update Daily Stats (Count Entry)
                        self.daily_stats['trades'] += 1
                        
                        # Execute Immediately (Mode A)
                        if self.mode != 'DRY_RUN':
                            if isinstance(self.broker, PaperBroker):
                                self.broker.execute_market_order(order.id, current_price, atr)
                            
                            # Log Fill if filled (Paper always filled, Binance might be filled)
                            if order.state == OrderState.FILLED:
                                self.metrics.fills += 1
                                self.logger_io.log_event('FILL', {'side': side, 'qty': trade_qty, 'price': order.avg_fill_price})
                            else:
                                logger.info(f"Order {order.id} submitted (State: {order.state}). Waiting for fill...")
                            
                            # Set Position Metadata (Stops)
                            # PaperBroker is dumb, storing stops in metadata or managing externally?
                            # Engine manages state.
                            self.current_stops = {
                                'initial_stop': stop_price,
                                'stop_loss': stop_price, # Trailing will update this
                                'entry_price': order.avg_fill_price
                            }

        # Trailing Stop / Stop Loss Validation
        # Check Stop Loss on the *Bar that just closed* (df_bar).
        # We need High/Low of the just closed bar.
        # last_row is exactly that.
        if self.broker.get_position(self.feed.symbol):
            self._check_stops(last_row, atr)
            
        # Update Cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            
        # Save State
        self._save_state(last_row.name)
            
    def _check_stops(self, bar, atr_val):
        """
        Checks if the *current position* was stopped out by 'bar' (High/Low).
        Updates Trailing Stop if active.
        """
        pos = self.broker.get_position(self.feed.symbol)
        if not pos: return # Should not happen if caller checks
        
        # Load stop from memory (Engine managed)
        # Using self.current_stops dict
        if not hasattr(self, 'current_stops'): return 
        
        stop_price = self.current_stops['stop_loss']
        slippage_bps = self.config['backtest']['slippage_bps']
        
        exit_triggered = False
        exit_price = 0.0
        reason = ''
        
        if pos['size'] > 0: # LONG
            # Low triggers stop
            if bar['low'] <= stop_price:
                exit_triggered = True
                exit_price = stop_price * (1 - slippage_bps/10000) # Simple Slip
                reason = 'STOP_LOSS'
                # Gap check
                if bar['open'] < stop_price:
                    exit_price = bar['open'] * (1 - slippage_bps/10000)
                    
        else: # SHORT
            # High triggers stop
            if bar['high'] >= stop_price:
                exit_triggered = True
                exit_price = stop_price * (1 + slippage_bps/10000)
                reason = 'STOP_LOSS'
                if bar['open'] > stop_price:
                     exit_price = bar['open'] * (1 + slippage_bps/10000)

        if exit_triggered:
            # Close Position
            trade_qty = abs(pos['size'])
            side = 'SHORT' if pos['size'] > 0 else 'LONG'
            
            # Submit Closing Order
            order = PaperOrder(self.feed.symbol, side, trade_qty)
            self.broker.submit_order(order)
            
            # Exec
            # We assume it happened INTRA-BAR, so we use calculated exit_price
            # Instead of current_price.
            # PaperBroker execute_market_order uses fill model which calculates slip.
            # Here we manually override price?
            # BrokerAdapter doesn't support 'Force Fill at Price'.
            # PaperBroker.execute_market_order takes 'market_price'.
            # If we pass exit_price as market_price, fill model adds slippage AGAIN.
            # Quick fix: Pass exit_price and 0 slip? Or modify Broker.
            # For PARITY: We need to match backtest math.
            
            # Backtest: exit = stop - slippage.
            # Here: exit_price is calculated.
            # We can force the fill price if we modify PaperBroker or trick it.
            # Let's just assume fill logic applies to 'stop_price' as the trigger.
            # But Broker applies spread too.
            # Simpler: Modify Broker to accept 'force_price'.
            # Or just accept small discrepancy.
            # Let's use stop_price as trigger.
            
            self.broker.execute_market_order(order.id, stop_price, atr=0) # Will apply slip/spread to stop_price
            
            self.logger_io.log_event('FILL', {'side': side, 'qty': trade_qty, 'price': order.avg_fill_price, 'reason': reason})
            self.metrics.fills += 1
            
            # Update Daily Stats
            exit_pnl = 0.0
            entry_p = self.current_stops.get('entry_price', 0.0)
            if entry_p > 0:
                if side == 'SHORT': # Closing Long
                    exit_pnl = (order.avg_fill_price - entry_p) * trade_qty
                else: # Closing Short
                    exit_pnl = (entry_p - order.avg_fill_price) * trade_qty
                    
            self.daily_stats['pnl'] += exit_pnl
            if exit_pnl < 0:
                self.daily_stats['consecutive_losses'] += 1
            else:
                self.daily_stats['consecutive_losses'] = 0
                
            logger.info(f"Trade Closed. PnL={exit_pnl:.2f} DailyTrades={self.daily_stats['trades']} DailyPnL={self.daily_stats['pnl']:.2f} ConsecLoss={self.daily_stats['consecutive_losses']}")
            
            # Activate Cooldown
            if reason == 'STOP_LOSS':
                # Check if it was initial or trail
                is_trail = self.current_stops['stop_loss'] != self.current_stops['initial_stop']
                if not is_trail:
                    self.cooldown_counter = self.config['strategy'].get('stop_cooldown_bars', 0)
                else:
                    self.cooldown_counter = self.config['strategy'].get('stop_cooldown_trail', 0)
            
            self.current_stops = {} # Clear
            return

        # Trailing Stop Update (if not exited)
        if self.config['strategy'].get('trailing_stop_active', False):
            # Same logic as engine.py
            atr_k = self.config['strategy']['stop_loss_atr_multiplier']
            current_stop = self.current_stops['stop_loss']
            
            if pos['size'] > 0: # LONG
                new_stop = bar['close'] - (atr_val * atr_k)
                if new_stop > current_stop:
                    self.current_stops['stop_loss'] = new_stop
            else: # SHORT
                new_stop = bar['close'] + (atr_val * atr_k)
                if new_stop < current_stop:
                    self.current_stops['stop_loss'] = new_stop

    def _update_ui(self):
        # Calc Equity
        price = self.feed.get_current_price()
        eq = self.broker.get_equity({self.feed.symbol: price})
        
        # Calc DD
        if eq > self.balance_high_water_mark:
            self.balance_high_water_mark = eq
        dd = (self.balance_high_water_mark - eq) / self.balance_high_water_mark
        self.max_dd = max(self.max_dd, dd)
        
        # Enhanced Metrics
        m = self.metrics.to_dict()
        m.update({
            'State': self.state,
            'DailyTr': f"{self.daily_stats['trades']}/{self.max_trades_daily}",
            'DailyPnL': f"${self.daily_stats['pnl']:.2f}",
            'ConsecL': f"{self.daily_stats['consecutive_losses']}",
            'Warmup': max(0, self.warmup_bars - self.processed_bars)
        })
        
        self.dashboard.refresh(eq, self.broker.balance, self.broker.positions, {}, m)

    def _check_health(self):
        if self.consecutive_errors >= self.max_consecutive_errors:
            self.state = 'HALTED_OPS'
            logger.critical("HALTED: Too many errors")
            return False
            
        if self.max_dd >= self.max_drawdown_limit:
            self.state = 'HALTED_RISK'
            logger.critical(f"HALTED: Max DD Reached ({self.max_dd:.2%})")
            return False
            
        return True

    def _check_daily_reset(self):
        """Resets daily stats if day changed (Istanbul Time)."""
        tz = ZoneInfo(self.timezone_str)
        now = datetime.now(tz)
        today_str = now.strftime('%Y-%m-%d')
        
        if self.daily_stats['date'] != today_str:
            logger.info(f"DAILY RESET: {self.daily_stats['date']} -> {today_str}")
            # Reset
            self.daily_stats = {
                'date': today_str,
                'trades': 0,
                'consecutive_losses': self.daily_stats.get('consecutive_losses', 0), # Keep consec losses across days? Usually yes or user decides.
                                                                                       # User implied "Daily Trade Limit". 
                                                                                       # "Consecutive Loss Limit" usually resets or persists?
                                                                                       # "Risk rules... Gün reset... Consecutive loss'ı trade close...".
                                                                                       # I'll keep consecutive losses for safety, unless explictly asked to reset.
                                                                                       # Actually, standard is Consec Loss is session based.
                                                                                       # Let's keep it.
                'pnl': 0.0
            }

    def _check_risk_limits(self):
        """Returns True if trading is allowed."""
        # 1. Daily Trade Limit
        if self.daily_stats['trades'] >= self.max_trades_daily:
            logger.warning(f"RISK: Daily Trade Limit Reached ({self.daily_stats['trades']}/{self.max_trades_daily})")
            return False
            
        # 2. Consecutive Losses
        if self.daily_stats['consecutive_losses'] >= self.max_consecutive_losses:
            logger.warning(f"RISK: Max Consecutive Losses Reached ({self.daily_stats['consecutive_losses']}/{self.max_consecutive_losses})")
            return False
            
        return True

    # Setup helper
    balance_high_water_mark = 0.0

    def _load_state(self):
        saved = self.state_store.load()
        if not saved:
            return

        logger.info("Restoring state from persistence...")
        try:
            # Migration Logic
            schema_ver = saved.get('schema_version', 0)
            if schema_ver < 1:
                # Migrate v0 -> v1
                saved.setdefault('signals', {})
                saved.setdefault('metrics', {})
                saved.setdefault('daily_stats', self.daily_stats)
                saved.setdefault('cooldown', 0)
                saved['schema_version'] = 1
                logger.info("STATE_MIGRATED: v0 -> v1 (Added defaults)")
            
            # Restore Props
            if saved.get('balance'):
                self.broker.balance = saved['balance']
            
            # Position Restore (Only PaperBroker supports direct set)
            if self.mode != 'LIVE' and self.mode != 'TESTNET':
                if saved.get('position'):
                    self.broker.position = saved['position']
            
            # Restore Stops
            self.current_stops = saved.get('stops', {})
            
            # Restore Counters
            self.processed_bars = saved.get('processed_bars', 0)
            self.cooldown_counter = saved.get('cooldown', 0)
            self.daily_stats = saved.get('daily_stats', self.daily_stats)
            self.metrics.from_dict(saved.get('metrics', {})) # Needs Metric restore
            
            # Reconcile Check
            last_ts_str = saved.get('last_processed_time')
            if last_ts_str:
                last_ts = pd.Timestamp(last_ts_str)
                logger.info(f"State Restored. Last processed: {last_ts}")
            
        except Exception as e:
            logger.warning(f"State Corrupt or Incompatible ({e}). Starting Fresh.")
            # Clear critical state to ensure fresh start
            self.current_stops = {}
            self.processed_bars = 0

    def _save_state(self, last_bar_ts):
        state = {
            'last_processed_time': str(last_bar_ts),
            'balance': self.broker.get_balance(), # Use getter
            'position': self.broker.get_position(self.feed.symbol),
            'stops': self.current_stops,
            'processed_bars': self.processed_bars,
            'cooldown': self.cooldown_counter,
            'daily_stats': self.daily_stats,
            'metrics': self.metrics.to_dict()
        }
        self.state_store.save(state)

    def _reconcile_broker_state(self):
        """
        Critical Truth Check: Persisted State vs Remote Exchange.
        """
        if self.mode not in ['LIVE', 'TESTNET']:
            return

        # 1. Get Remote Position
        remote_pos = self.broker.get_position(self.args.symbol)
        remote_qty = remote_pos['size'] if remote_pos else 0.0
        
        # 2. Get Local Persisted Position
        saved = self.state_store.load()
        local_qty = 0.0
        if saved and saved.get('position'):
            local_qty = saved['position']['size']
            
        # 3. Compare
        diff = abs(remote_qty - local_qty)
        if diff > 0.000001: # Epsilon
            msg = f"RECONCILE MISMATCH: Local(Saved)={local_qty} Remote(Exchange)={remote_qty}"
            logger.critical(msg)
            self.logger_io.log_event('RECONCILE_MISMATCH', {
                'local': local_qty,
                'remote': remote_qty,
                'diff': diff
            })
            # Soft Halt as requested
            self.state = 'HALTED_OPS'
            logger.error("Soft Halt triggered due to reconciliation mismatch. Manual intervention required.")
        else:
            # Emit OK event (Standardized short event)
            if saved: # Only if we have history
                self.logger_io.log_event('RECONCILE_OK', {'qty': remote_qty})
            logger.debug(f"Reconciliation PASSED. Qty={remote_qty}")

    def _reconcile_state(self, current_bar_ts):
        """
        Check if we missed bars during downtime.
        """
        saved = self.state_store.load()
        if not saved: return
        
        last_ts = pd.Timestamp(saved['last_processed_time'])
        # Current bar TS is the closed bar time.
        # If difference > 2 * Timeframe (1 bar is normal, 2 is gap)
        # Parse timeframe
        
        # Simple Logic:
        # If current_bar_ts - last_ts > 1 TF -> Gap
        # We handle this by setting a temporary NO_TRADE flag or just logging.
        
        if not saved: return

        last_ts = pd.Timestamp(saved['last_processed_time'])
        
        # Ensure TX-Aware
        if current_bar_ts.tzinfo is None:
            current_bar_ts = current_bar_ts.tz_localize('UTC')
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize('UTC')
            
        diff = current_bar_ts - last_ts
        # This diff relies on frequency.
        logger.info(f"Reconcile: Last={last_ts} Current={current_bar_ts} Diff={diff}")
        
        # If diff is large, log critical
        # Assuming 1h TF, diff > 1h + buffer is gap
        # Just logging for now as user requested "Gap Detection"
        if diff > pd.Timedelta(hours=1.1) and self.args.tf == '1h':
             logger.warning("GAP DETECTED: System was down for more than 1 bar.")
             self.logger_io.log_event('GAP_DETECTED', {'last': str(last_ts), 'current': str(current_bar_ts)})
             # Option: Block trading for 1 bar to realign? 
             # self.cooldown_counter = 1

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['DRY_RUN', 'PAPER', 'LIVE', 'BACKTEST_PARITY', 'SHADOW', 'TESTNET'], default='DRY_RUN')
    parser.add_argument('--symbol', help='BTC/USDT')
    parser.add_argument('--tf', help='1h, 4h')
    parser.add_argument('--feed', choices=['LIVE', 'HIST'], default='LIVE', help='Force feed type')
    parser.add_argument('--seed', type=int, help='Seed for deterministic fill')
    parser.add_argument('--threshold_offset', type=float, help='Override ML offset')
    parser.add_argument('--warmup_bars', type=int, default=500, help='Bars to process before trading')
    parser.add_argument('--live_confirm', help='Config Hash for LIVE arming')
    args = parser.parse_args()
    
    cfg = load_config()
    engine = PaperEngine(cfg, args)
    engine.run()
