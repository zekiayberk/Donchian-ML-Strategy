
import pandas as pd
import numpy as np

class PerformanceMetrics:
    @staticmethod
    def calculate(trades_df, equity_df, initial_capital):
        """
        Detaylı performans metriklerini hesaplar.
        """
        if trades_df.empty:
            return {"Total Trades": 0, "Final Equity": initial_capital, "BnH Return": 0.0}
            
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Drawdown
        equity_series = equity_df['equity']
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade Istatistikleri
        win_trades = trades_df[trades_df['pnl'] > 0]
        loss_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(win_trades) / len(trades_df)
        avg_win = win_trades['pnl'].mean() if not win_trades.empty else 0
        avg_loss = loss_trades['pnl'].mean() if not loss_trades.empty else 0
        profit_factor = abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum()) if abs(loss_trades['pnl'].sum()) > 0 else float('inf')
        
        # CAGR (Basit: Toplam gün sayısına göre)
        days = (equity_df['time'].iloc[-1] - equity_df['time'].iloc[0]).days
        if days > 0:
            cagr = (final_equity / initial_capital) ** (365 / days) - 1
        else:
            cagr = 0
            
        # Exit Reasons
        initial_stops = len(trades_df[trades_df['exit_reason'] == 'INITIAL_STOP'])
        trail_stops = len(trades_df[trades_df['exit_reason'] == 'TRAIL_STOP'])
            
        return {
            "Initial Capital": initial_capital,
            "Final Equity": final_equity,
            "Total Return (%)": total_return * 100,
            "CAGR (%)": cagr * 100,
            "Max Drawdown (%)": max_drawdown * 100,
            "Total Trades": len(trades_df),
            "Win Rate (%)": win_rate * 100,
            "Profit Factor": profit_factor,
            "Avg Win": avg_win,
            "Avg Loss": avg_loss,
            "Avg Trade": trades_df['pnl'].mean(),
            "INITIAL_STOP Count": initial_stops,
            "TRAIL_STOP Count": trail_stops
        }

from rich.console import Console
from rich.table import Table

def print_report(metrics, trades_df):
    console = Console()
    
    table = Table(title="Backtest Sonuçları", show_header=True, header_style="bold magenta")
    table.add_column("Metrik", style="cyan")
    table.add_column("Değer", style="green")
    
    for k, v in metrics.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.2f}")
        else:
            table.add_row(k, str(v))
            
    console.print(table)
    console.print(f"[bold]Toplam Trade Sayısı:[/bold] {len(trades_df)}")
