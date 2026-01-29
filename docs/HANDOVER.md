# Production Handover & Operations Manual

## 1. Safety & Hard Gate
The bot is protected by a **Hard Gate**. It will NOT execute in `LIVE` mode unless explicitly armed.

### Launching Canary (First Trade)
1. Set Environment Variables:
   ```bash
   export BINANCE_TESTNET_KEY="your_key"
   export BINANCE_TESTNET_SECRET="your_secret"
   export LIVE_TRADING="YES_I_UNDERSTAND"
   ```
2. Get the Config Hash (Dry Run):
   ```bash
   python3 live/run_paper.py --mode LIVE
   ```
   *It will fail and print `RealHash=...`*
3. Run with Confirmation:
   ```bash
   python3 live/run_paper.py --mode LIVE --live_confirm <HASH_FROM_STEP_2>
   ```

## 2. Ramping Plan (Scaling)

| Stage | Criteria | Risk Per Trade | Max Daily Loss | Actions |
|-------|----------|----------------|----------------|---------|
| **Stage 0 (Canary)** | First 30 Days / 20 Trades | 0.10% | 0.5% | 1 Trade/Day limit. Verify Fees & Latency. |
| **Stage 1** | Post-Canary Success | 0.15% | 0.5% | Increase position size. Keep daily limit. |
| **Stage 2** | Stable (>60 Days) | 0.20% | 1.0% | Normal operation. |

**To Advance Stage**:
1. Review `reports/daily_*.md`.
2. Update `config.yaml` risk parameters.
3. Restart bot (New Config Hash required).

## 3. Emergency Playbook

### Scenario A: Bot Crashes / Loop
1. Check process status: `pm2 status` or `ps aux | grep python`
2. View Logs: `tail -f events.jsonl` or `pm2 logs`
3. If critical error (`LIVE_ARM_FAILED`, `RECONCILE_MISMATCH`):
   - **Do not restart blindly.**
   - Fix the issue (e.g., sync clock, check Binance API status).
   - Re-arm with new Hash if config changed.

### Scenario B: Stuck Position (Bot Halted)
If Bot halts (`HALTED_OPS`) but you have an open position on Binance:
1. **Login to Binance Mobile App / Web** immediately.
2. Manually Close the position.
3. Update `paper_state.json` or delete it to reset state (Only after manual close!).

## 4. Supervisor Setup (PM2)
Recommended for 24/7 uptime on Linux/Mac:

```bash
# Install PM2
npm install pm2 -g

# Start Bot
pm2 start live/run_paper.py --name "donchian-bot" --interpreter python3 -- \
  --mode LIVE --live_confirm <HASH>
```

### 5. Verification (Shadow Mode)
**Objective**: Confirm system stability with real data but ZERO execution risk.

1. **Clear Safety Lock**: `unset LIVE_TRADING`
2. **Run Shadow**:
   ```bash
   PYTHONPATH=. python3 live/run_paper.py --mode SHADOW --feed LIVE
   ```
3. **Verify Logs (`tail -f events.jsonl`)**:
   - `BOOT_SEAL`: usage `SHADOW`
   - `SHADOW_SAFETY_OK`: Checks passed.
   - `LiveFeed`: "Seed Data OK"
   - `SHADOW_ORDER`: Signals generated (simulated).

### 6. Troubleshooting
- **State Load Error**: System will auto-migrate or reset. Check logs for `STATE_MIGRATED`.
- **Feed Waiting**: System retries 5 times. If stuck, check internet connection.

## 7. Daily Reporting
A report is generated daily from `events.jsonl`.
Run manually:
```bash
python3 live/reporter.py
```
Output: `reports/daily_YYYY-MM-DD.md`
