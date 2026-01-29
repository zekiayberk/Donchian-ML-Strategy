# Donchian Breakout + ML Filtering Strategy


## Overview
This project is an advanced algorithmic trading system designed for cryptocurrency markets (primarily Binance Futures). It combines a classic **Donchian Breakout** strategy with **ATR-based Trailing Stops** and an optional **Machine Learning (LightGBM/RandomForest) signal filter** to reduce false breakouts.

The system is built with robustness in mind, featuring a full **Walk-Forward Optimization (WFO)** engine and a local **Shadow/Paper trading** mode for live validation.

## Key Features
- **Dynamic Breakout Detection**: Uses Donchian Channels for entry signals.
- **Adaptive Risk Management**: ATR-based stop loss and trailing stop mechanisms.
- **ML Signal Filtering**: Probabilistic filtering using time-series features to avoid low-probability setups.
- **Walk-Forward Optimization**: Continuous parameter/model validation to prevent overfitting.
- **Paper/Shadow Trading**: Run the bot against live data without risking real capital.
- **Live Implementation**: CCXT-based execution for Binance Futures.
- **Comprehensive Logging**: Detailed event tracking in `.jsonl` format for later analysis.

## Setup & Installation

1. **Python Version**: Python 3.11+ is recommended.
2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/donchian-ml-strategy.git
   cd donchian-ml-strategy
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables**:
   Copy the example file and fill in your API credentials:
   ```bash
   cp .env.example .env
   ```

## Usage Guide

### 1. Backtesting
Run the strategy on historical data with default settings:
```bash
python run_backtest.py
```
Custom symbols and timeframes:
```bash
python run_backtest.py --symbol ETHUSDT --tf 4h --start 2023-01-01 --end 2024-01-01
```

### 2. Optimization (Walk-Forward)
Perform parameter sweeping and ML model training:
```bash
python run_wfo.py --symbol BTCUSDT --tf 1h
```

### 3. Live / Paper Trading
Run the bot in shadow mode (real data, virtual orders):
```bash
python live/run_paper.py --mode SHADOW --feed LIVE
```
To enable LIVE trading, you must set `LIVE_TRADING=YES_I_UNDERSTAND` in your `.env` and use the `--mode LIVE` flag.

## Project Structure
- `src/`: Core logic (ML modules, utility functions).
- `indices/`: Technical indicator implementations (ATR, Donchian).
- `strategy/`: Signal generation logic.
- `backtest/`: High-performance backtesting engine.
- `optimization/`: WFO and parameter sweep engines.
- `live/`: Live data feeds and broker execution adapters.
- `data/`: Data fetching and storage management.
- `config.yaml`: Central configuration for all strategy parameters.

## Strategy Details
- **Entry**: Price breaks above/below the N-period Donchian Channel.
- **Exit**: ATR-based trailing stop (Dynamic).
- **ML Filter**: A classifier trained on historical breakouts to predict the probability of a "successful" move.

## Disclaimer
This software is for educational purposes only. Do not trade with money you cannot afford to lose. Past performance does not guarantee future results. It is not investment advice

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


