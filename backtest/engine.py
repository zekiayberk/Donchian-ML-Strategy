
import pandas as pd
import numpy as np
import collections
from .position import Position, Trade
from indicators.donchian import calculate_donchian_channel
from indicators.atr import calculate_atr
from strategy.signals import SignalGenerator
from src.ml.inference import MLEngine
from src.ml.features import engineering_features, get_feature_row_at

class BacktestEngine:
    def __init__(self, df, config, ml_enabled=False, threshold_offset=0.0):
        """
        :param df: OHLCV DataFrame
        :param config: Dict (strateji ve risk parametreleri)
        :param ml_enabled: ML filtresi aktif mi?
        :param threshold_offset: ML olasılık eşiği için kaydırma (Sensitivity analizi için)
        """
        self.raw_df = df
        self.config = config
        self.ml_enabled = ml_enabled
        self.threshold_offset = threshold_offset
        self.ml_engine = None
        
        if self.ml_enabled:
            # Multi-fold modelleri yükle
            # Config'i geçiriyoruz ki Guard state init edilebilsin
            self.ml_engine = MLEngine("models", threshold_offset=self.threshold_offset, config=self.config)
            
        self.trades: list[Trade] = []
        self.equity_curve = []
        self.current_position = None
        self.equity = config['backtest']['initial_capital']
        self.initial_capital = config['backtest']['initial_capital']
        
        # Parametreleri al
        self.donchian_n = config['strategy']['donchian_period']
        self.atr_n = config['strategy']['atr_period']
        self.atr_k = config['strategy']['stop_loss_atr_multiplier']
        self.risk_per_trade = config['risk']['risk_per_trade']
        self.slippage_bps = config['backtest']['slippage_bps']
        self.fee_maker = config['backtest']['commission_futures_maker'] 
        self.fee_taker = config['backtest']['commission_futures_taker'] 
        
        # Cooldown
        self.stop_cooldown_bars = config['strategy'].get('stop_cooldown_bars', 0)
        self.cooldown_counter = 0 # 0 ise işlem yapılabilir
        
        # İstatistikler
        self.stats = {
            'signals_total': 0,
            'signals_total_in_oos': 0,
            'signals_taken': 0,
            'signals_skipped': 0,
            'folds': {}, # fold_id -> {total, taken, skipped, all_probs}
            'forward': {
                'total': 0,         # ML evaluate fonksiyonuna giren
                'taken': 0, 
                'skipped': 0, 
                'probs': [],
                'thresholds': [],
                'status_counts': collections.defaultdict(int),
                'prob_counts': collections.defaultdict(int)
            }
        }
        
        
        # Blocked Execution Tracking
        self.ml_allowed = 0
        self.ml_skipped = 0
        self.entry_opened = 0
        self.entry_blocked = 0
        self.open_position_at_end = 0
        self.entry_block_reasons = collections.Counter()
        self.blocked_events = []
        
        # Hazırlık
        self.prepare_data()

    def prepare_data(self):
        """İndikatörleri hesapla"""
        df = self.raw_df.copy()
        df = calculate_donchian_channel(df, self.donchian_n)
        df = calculate_atr(df, self.atr_n)
        # Config'i geçir (Filtreler için)
        df = SignalGenerator.generate_signals(df, self.config)
        
        # ML Features (Eğer ML aktifse veya eğitim için gerekiyorsa)
        if self.ml_enabled:
            print("Feature Engineering yapılıyor...")
            df = engineering_features(df)
            
        signal_counts = df['entry_signal'].value_counts().to_dict()
        print(f"\n[DEBUG] Raw Signals Generated: {signal_counts}")
        if 1 not in signal_counts and -1 not in signal_counts:
            print("[WARNING] Hiçbir sinyal üretilemedi! Config ayarlarını (ADX, Breakout Strength) kontrol edin.")
            
        self.df = df
        
        if self.ml_enabled:
            pass # Guard state artık MLEngine içinde yönetiliyor

    def run(self):
        """
        Event-driven (bar-by-bar) backtest döngüsü.
        """
        # Datetime index ise kolona çevir
        work_df = self.df.copy()
        if isinstance(work_df.index, pd.DatetimeIndex):
            work_df.reset_index(inplace=True)
            
        data = work_df.to_dict('records') # Liste olarak al, daha hızlı erişim
        
        for i in range(len(data)):
            # Cooldown Sayacını Azalt
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                
            # İlk N bar (lookback) atla
            if i < max(self.donchian_n, self.atr_n):
                continue
                
            bar = data[i]
            current_time = bar['datetime']
            open_price = bar['open']
            high_price = bar['high']
            low_price = bar['low']
            close_price = bar['close']
            atr_value = bar['atr']
            entry_signal = bar['entry_signal']
            
            # --- Equity Curve Güncelleme (Mark-to-Market) ---
            current_equity = self.equity
            if self.current_position:
                unrealized_pnl = 0
                if self.current_position.direction == 'LONG':
                    unrealized_pnl = (close_price - self.current_position.entry_price) * self.current_position.qty
                else:
                    unrealized_pnl = (self.current_position.entry_price - close_price) * self.current_position.qty
                current_equity += unrealized_pnl
            
            self.equity_curve.append({
                'time': current_time, 
                'equity': current_equity,
                'drawdown': 0.0 # Sonra hesaplanacak
            })

            # --- Pozisyon Kontrolü ve Stop Loss ---
            if self.current_position:
                # Check for Stop Loss HIT INTRA-BAR
                stop_hit = False
                exit_price = 0.0
                
                if self.current_position.direction == 'LONG':
                    if low_price <= self.current_position.stop_loss:
                        stop_hit = True
                        # Slippage uygula
                        slippage = self.current_position.stop_loss * (self.slippage_bps / 10000)
                        exit_price = self.current_position.stop_loss - slippage
                        # Gap durumu
                        if open_price < self.current_position.stop_loss:
                            exit_price = open_price - slippage 
                        
                elif self.current_position.direction == 'SHORT':
                    if high_price >= self.current_position.stop_loss:
                        stop_hit = True
                        slippage = self.current_position.stop_loss * (self.slippage_bps / 10000)
                        exit_price = self.current_position.stop_loss + slippage
                        if open_price > self.current_position.stop_loss:
                            exit_price = open_price + slippage

                if stop_hit:
                    # Reason belirle
                    reason = 'INITIAL_STOP'
                    if self.current_position.stop_loss != self.current_position.initial_stop_loss:
                        reason = 'TRAIL_STOP'
                    
                    # Dinamik Cooldown
                    if reason == 'INITIAL_STOP':
                        self.cooldown_counter = self.stop_cooldown_bars
                    else:
                        trail_cd = self.config['strategy'].get('stop_cooldown_trail', 3)
                        self.cooldown_counter = trail_cd
                        
                    self.close_position(exit_price, current_time, reason)
                    continue # Pozisyon kapandı

                # --- Trailing Stop Update ---
                # Sadece bar kapanışında güncelle (conservative)
                self.current_position.update_trailing_stop(close_price, atr_value, self.atr_k)

            # --- Sinyal Kontrolü ---
            # Eğer pozisyon yoksa VE cooldown yoksa yeni giriş bak
            if not self.current_position and self.cooldown_counter == 0:
                if entry_signal != 0:
                    # ML Metadata default values
                    ml_res = {'is_allowed': True, 'prob': 1.0, 'fold': -1, 'status': 'disabled'}
                    
                    # ML Filtresi
                    is_allowed = True
                    if self.ml_enabled and self.ml_engine:
                            # Feature satırı hazırla (Dict-based, import riski yok)
                            try:
                                # Dataframe'den dict al
                                row_dict = self.df.iloc[i].to_dict()
                                
                                # Manuel hesapla ve ekle
                                c = row_dict.get('close')
                                atr_v = row_dict.get('atr', 0)
                                if entry_signal == 1:
                                    upper = row_dict.get('donchian_upper', 0)
                                    row_dict['breakout_strength'] = (c - upper) / atr_v if atr_v and atr_v > 0 else 0
                                elif entry_signal == -1:
                                    lower = row_dict.get('donchian_lower', 0)
                                    row_dict['breakout_strength'] = (lower - c) / atr_v if atr_v and atr_v > 0 else 0
                                else:
                                    row_dict['breakout_strength'] = 0.0
                                    
                                row_dict['direction'] = int(entry_signal)
                                row_dict['timestamp'] = current_time
                                
                                # DEBUG: Keys kontrol
                                # print(f"[DEBUG KEYS] BS={'breakout_strength' in row_dict} DIR={'direction' in row_dict} LEN={len(row_dict)}")
                                
                                # Inference (Returns dict)
                                ml_res = self.ml_engine.predict(pd.DataFrame([row_dict]))
                                
                                # --- DEBUG LOGGING (User Request) ---
                                # Her sinyalde (oos_test olsun olmasın) detayları bas
                                # --- DEBUG LOGGING Fix ---
                                feat_cols = ml_res.get('feature_cols', [])
                                # self.df yerine row_dict kullanıyoruz çünkü dynamic featurelar orada
                                X_row = pd.Series(row_dict)[feat_cols] if feat_cols and 'row_dict' in locals() else pd.Series()
                                nan_count = X_row.isna().sum() if not X_row.empty else "N/A"
                                std_val = X_row.std() if not X_row.empty else 0.0
                                unique_val = X_row.nunique() if not X_row.empty else 0
                                
                                fold_info = f"fold_{ml_res['fold']:02d}" if ml_res['fold'] != -1 else "N/A"
                                
                                print(f"\n[DEBUG ML] Idx: {i} | Time: {current_time} | Status: {ml_res['status']} | Fold: {fold_info}")
                                print(f"  > Index Check: Loop_Idx={i} | DF_Index={self.df.index[i]}")
                                print(f"  > Stats: NaN={nan_count} | Std={std_val:.6f} | Unique={unique_val}")
                                print(f"  > Result: Prob={ml_res['prob']:.4f} | Target_T={ml_res.get('used_threshold', 0.5):.2f}")
                                if not X_row.empty:
                                    print(f"  > Top Features: {X_row.head(5).to_dict()}")
                                # ------------------------------------

                                # Fold-Aware Gating Kuralı:
                                # Sadece status == 'oos_test' ise filtreleme yapıyoruz (Gerçek WFO OOS Equity için)
                                if ml_res['status'] == 'oos_test':
                                    fold_id = f"fold_{ml_res['fold']:02d}"
                                    if fold_id not in self.stats['folds']:
                                        self.stats['folds'][fold_id] = {'total': 0, 'taken': 0, 'skipped': 0, 'all_probs': [], 'threshold': ml_res['used_threshold']}
                                    
                                    self.stats['signals_total_in_oos'] += 1
                                    self.stats['folds'][fold_id]['total'] += 1
                                    self.stats['folds'][fold_id]['all_probs'].append(ml_res['prob'])

                                    # ML Kararı (Conditional Guard via MLEngine)
                                    fold_id = f"fold_{ml_res['fold']:02d}"
                                    base_t = ml_res['used_threshold'] - self.threshold_offset
                                    
                                    is_allowed, tag, used_t, meta = self.ml_engine.decide_with_guard(
                                        prob=ml_res['prob'],
                                        fold=fold_id,
                                        base_threshold=base_t,
                                        offset=self.threshold_offset
                                    )
                                    
                                    if is_allowed:
                                        if tag == "GUARD":
                                            status_str = "ALLOWED (GUARD)"
                                            # Debug log
                                            if self.ml_engine.guard_debug:
                                                print(f"    [GUARD ACTIVATED] {fold_id} | Prob={ml_res['prob']:.4f} >= GuardT={used_t:.4f}")
                                        else:
                                            status_str = "ALLOWED"
                                    else:
                                        status_str = "SKIPPED"
                                        # logic: guard aktif ama prob yetmediyse log basabiliriz
                                        if "low_base_rate" in str(meta.get("reason", "")):
                                             if self.ml_engine.guard_debug:
                                                 print(f"    [GUARD SKIP]      {fold_id} | Prob={ml_res['prob']:.4f} < GuardT={meta.get('guard_t', 0):.4f}")
                                            
                                    print(f"  > Final Decision: {status_str} (OOS Mode)\n")

                                    if not is_allowed:
                                        self.stats['signals_skipped'] += 1
                                        self.stats['folds'][fold_id]['skipped'] += 1
                                    else:
                                        self.stats['signals_taken'] += 1
                                        self.stats['folds'][fold_id]['taken'] += 1
                                elif ml_res['status'] == 'outside_test_range' and ml_res.get('source', '') != 'default_config':
                                    # Forward Test or Future: Apply ML filter but without Stats/Guard
                                    is_allowed = ml_res['is_allowed']
                                    status_str = "ALLOWED" if is_allowed else "SKIPPED"
                                    
                                    # Forward Stats Update
                                    self.stats['forward']['total'] += 1
                                    self.stats['forward']['probs'].append(ml_res['prob'])
                                    self.stats['forward']['thresholds'].append(ml_res['used_threshold'])
                                    self.stats['forward']['status_counts'][ml_res['status']] += 1
                                    
                                    # Prob histogram
                                    p_round = round(ml_res['prob'], 4)
                                    self.stats['forward']['prob_counts'][p_round] += 1
                                    
                                    if is_allowed:
                                        self.stats['forward']['taken'] += 1
                                    else:
                                        self.stats['forward']['skipped'] += 1
                                        
                                    print(f"  > Final Decision: {status_str} (Forward/Non-OOS Mode)\n")
                                else:
                                    # Test aralığı dışındaysa (train/valid/no_model) ML gating uygulama
                                    print(f"  > Final Decision: ALLOWED (Non-OOS Mode: {ml_res['status']})\n")
                                    is_allowed = True
                                    
                            except Exception as e:
                                print(f"ML Inference hatası (Bar {i}): {e}")
                            
                            except Exception as e:
                                print(f"ML Inference hatası (Bar {i}): {e}")
                            
                            
                    if is_allowed:
                        self.ml_allowed += 1
                        
                        # --- EXECUTION BLOCK CHECK ---
                        block_reason = self._check_execution_block(entry_signal, data[i]['close'])
                        
                        if block_reason:
                            self.entry_blocked += 1
                            self.entry_block_reasons[block_reason] += 1
                            
                            self.blocked_events.append({
                                'timestamp': data[i]['datetime'],
                                'signal': entry_signal,
                                'reason': block_reason,
                                'ml_prob': ml_res.get('prob', -1),
                                'ml_threshold': ml_res.get('used_threshold', -1)
                            })
                            # print(f"[EXEC BLOCK] Signal Allowed by ML but Blocked: {block_reason}")
                            continue # Skip open_position
                    
                        # Giriş fiyatı: Mode A (Next Bar Open)
                        if i + 1 < len(data):
                            next_open = data[i+1]['open']
                            
                            # Slippage
                            slippage_amount = next_open * (self.slippage_bps / 10000)
                            
                            if entry_signal == 1:
                                entry_price = next_open + slippage_amount
                                opened, reason = self.open_position('LONG', entry_price, data[i+1]['datetime'], atr_value, 
                                                  ml_active=(ml_res['status']=='oos_test'), 
                                                  ml_prob=ml_res['prob'], 
                                                  ml_fold=ml_res['fold'])
                            elif entry_signal == -1:
                                entry_price = next_open - slippage_amount
                                opened, reason = self.open_position('SHORT', entry_price, data[i+1]['datetime'], atr_value,
                                                  ml_active=(ml_res['status']=='oos_test'), 
                                                  ml_prob=ml_res['prob'], 
                                                  ml_fold=ml_res['fold'])
                            
                            if opened:
                                self.entry_opened += 1
                            else:
                                self.entry_blocked += 1
                                self.entry_block_reasons[reason] += 1
                                self.blocked_events.append({
                                    'timestamp': data[i+1]['datetime'],
                                    'signal': entry_signal,
                                    'reason': reason,
                                    'ml_prob': ml_res.get('prob'),
                                    'ml_threshold': ml_res.get('used_threshold')
                                })
                        else:
                             # Last bar case - cannot execute next open
                             self.entry_blocked += 1
                             reason = "END_OF_DATA"
                             self.entry_block_reasons[reason] += 1
                             self.blocked_events.append({
                                'timestamp': data[i]['datetime'],
                                'signal': entry_signal,
                                'reason': reason,
                                'ml_prob': ml_res.get('prob'),
                                'ml_threshold': ml_res.get('used_threshold')
                             })
                    else:
                        self.ml_skipped += 1

        # Sonuçları yazdır
        if self.ml_enabled:
            # Check open position at end
            self.open_position_at_end = 1 if (self.current_position is not None) else 0
            self.print_ml_stats()

    def open_position(self, direction, price, time, atr, ml_active=False, ml_prob=1.0, ml_fold=-1):
        # Position Sizing
        risk_per_share = self.atr_k * atr
        
        if risk_per_share <= 0:
            return False, "INVALID_RISK_PER_SHARE"
        
        # Stop Price
        if direction == 'LONG':
            stop_loss = price - risk_per_share
        else:
            stop_loss = price + risk_per_share
            
        # Equity'nin %R'si kadar risk al
        risk_amount = self.equity * self.risk_per_trade
        
        if risk_per_share == 0:
            qty = 0
        else:
            qty = risk_amount / risk_per_share
            
        # Pozisyon açılış komisyonu
        commission = (price * qty) * self.fee_taker
        self.equity -= commission
        
        pos = Position(
            symbol="TEST",
            direction=direction,
            entry_time=time,
            entry_price=price,
            qty=qty,
            stop_loss=stop_loss,
            ml_active=ml_active,
            ml_prob=ml_prob,
        )
        self.current_position = pos
        return True, "FILLED"

    def close_position(self, price, time, reason):
        pos = self.current_position
        
        # PnL
        if pos.direction == 'LONG':
            pnl_gross = (price - pos.entry_price) * pos.qty
        else:
            pnl_gross = (pos.entry_price - price) * pos.qty
            
        # Kapanış komisyonu
        commission = (price * pos.qty) * self.fee_taker
        pnl_net = pnl_gross - commission
        
        self.equity += pnl_net
        
        # Trade kaydı
        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=time,
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=price,
            qty=pos.qty,
            pnl=pnl_net,
            pnl_percent=(pnl_net / (pos.entry_price * pos.qty)) * 100, 
            exit_reason=reason,
            commission=commission,
            ml_active=pos.ml_active,
            ml_prob=pos.ml_prob,
            ml_fold=pos.ml_fold
        )
        self.trades.append(trade)
        self.current_position = None

    def _check_execution_block(self, signal, price_approx):
        """
        Checks if execution is blocked by rules (Position, Risk, Cooldown).
        Returns reason code (str) or None (Allowed).
        """
        # 1. Position already open? (Wait, logic structure ensures this is mostly None here, 
        # but for future parallel reqs/multi-symbol, good to have)
        if self.current_position is not None:
            return "POSITION_OPEN"

        # 2. Cooldown
        if self.cooldown_counter > 0:
            return "COOLDOWN_ACTIVE"

        # 3. Max Risk / Daily Loss (Simulation)
        # E.g. risk per trade > available cash * risk_limit
        # Minimal simulation: if equity dropped too much today (complex to track 'today' in this simple loop)
        # Placeholder for simple margin check:
        if self.equity <= 0:
            return "BANKRUPT"
            
        # 4. Spread / Slippage filters (Future)
        
        return None

    def get_results(self):
        return pd.DataFrame([t.__dict__ for t in self.trades]), pd.DataFrame(self.equity_curve)

    def print_ml_stats(self):
        print("\n" + "="*40)
        print("      ML SELECTIVITY REPORT (OOS)")
        print("="*40)
        print(f"Total Signals in OOS: {self.stats['signals_total']}")
        print(f"Signals Taken:        {self.stats['signals_taken']}")
        print(f"Signals Skipped:      {self.stats['signals_skipped']}")
        
        if self.stats['signals_total'] > 0:
            skip_rate = (self.stats['signals_skipped'] / self.stats['signals_total']) * 100
            print(f"Overall Skip Rate:    {skip_rate:.2f}%")
        
        print("-" * 65)
        header = f"{'FOLD':<10} | {'TOTAL':<5} | {'TAKE':<5} | {'SKIP':<5} | {'T':<5} | {'MIN':<6} | {'MEAN':<6} | {'MAX':<6}"
        print(header)
        print("-" * 65)
        for fid in sorted(self.stats['folds'].keys()):
            f = self.stats['folds'][fid]
            probs = f['all_probs']
            if probs:
                p_min, p_mean, p_max = np.min(probs), np.mean(probs), np.max(probs)
                print(f"{fid:<10} | {f['total']:<5} | {f['taken']:<5} | {f['skipped']:<5} | {f['threshold']:<5.2f} | {p_min:<6.4f} | {p_mean:<6.4f} | {p_max:<6.4f}")
            else:
                print(f"{fid:<10} | {f['total']:<5} | {f['taken']:<5} | {f['skipped']:<5} | {f['threshold']:<5.2f} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6}")
        print("="*65 + "\n")

        # --- FORWARD / NON-OOS REPORT ---
        ft = self.stats.get('forward', {'total':0})
        if ft['total'] > 0:
            print(f"\n========================================")
            print(f"   FORWARD / NON-OOS SELECTIVITY REPORT")
            print(f"========================================")
            print(f"Total Signals Evaluated: {ft['total']}")
            print(f"Signals Taken:           {ft['taken']}")
            print(f"Signals Skipped:         {ft['skipped']}")
            skip_rate = (ft['skipped']/ft['total'])*100 if ft['total']>0 else 0
            print(f"Skip Rate:               {skip_rate:.2f}%")
            
            if ft['probs']:
                probs = ft['probs']
                thresholds = ft['thresholds']
                p_min, p_mean, p_max = np.min(probs), np.mean(probs), np.max(probs)
                print(f"Prob Stats:   Min={p_min:.4f} | Mean={p_mean:.4f} | Max={p_max:.4f}")
                
                t_min, t_mean, t_max = np.min(thresholds), np.mean(thresholds), np.max(thresholds)
                print(f"Target T Stats: Min={t_min:.4f} | Mean={t_mean:.4f} | Max={t_max:.4f}")

                # Execution Block Report (New Logic)
                print(f"ML Allowed:             {self.ml_allowed}")
                print(f"Entry Opened (FILLED):  {self.entry_opened}")
                print(f"Entry Blocked:          {self.entry_blocked}")
                print(f"Closed Trades:          {len(self.trades)}")
                print(f"Open Position At End:   {self.open_position_at_end}")
                
                not_executed = self.ml_allowed - self.entry_opened
                
                if not_executed > 0:
                    print(f"\nNot Executed:           {not_executed} (should equal Entry Blocked)")
                    
                    if self.entry_blocked > 0:
                        print("\nBlocked Reason Breakdown:")
                        for reason, cnt in self.entry_block_reasons.most_common():
                            print(f"  {reason}: {cnt}")
                            
                        if self.blocked_events:
                            last = self.blocked_events[-1]
                            print("\nLast Blocked Event:")
                            print(f"  Time={last['timestamp']} | Signal={last['signal']} | Reason={last['reason']} | Prob={last.get('ml_prob'):.4f}")

                else:
                    print("\nAll ML-allowed signals were executed.")
                    if self.open_position_at_end > 0:
                        print(f"  (Note: 1 Open Position at End)")

                print("\nFrequency Distribution (Top 5 Probs):")
                sorted_probs = sorted(ft['prob_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
                for p, c in sorted_probs:
                    print(f"  Prob {p:.4f}: {c} times")

            print("========================================\n")
