
import matplotlib.pyplot as plt
import pandas as pd

def plot_results(equity_df, trades_df, filename="backtest_result.png"):
    """
    Equity curve ve drawdown grafiğini çizer ve kaydeder.
    """
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Equity Curve
    plt.subplot(2, 1, 1)
    plt.plot(equity_df['time'], equity_df['equity'], label='Equity', color='blue')
    plt.title('Equity Curve')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Subplot 2: Drawdown
    plt.subplot(2, 1, 2)
    equity_series = equity_df['equity']
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    
    plt.plot(equity_df['time'], drawdown, label='Drawdown', color='red')
    plt.fill_between(equity_df['time'], drawdown, 0, color='red', alpha=0.3)
    plt.title('Drawdown')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Grafik kaydedildi: {filename}")
