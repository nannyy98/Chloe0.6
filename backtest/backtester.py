"""
Backtesting Module for Chloe AI
Tests trading strategies on historical data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import seaborn as sns

logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtesting engine for evaluating trading strategies
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.results = {}
        
    def run_backtest(self, data: pd.DataFrame, signals: pd.Series, 
                     transaction_cost: float = 0.001) -> Dict:
        """
        Run backtest on historical data with given signals
        
        Args:
            data: DataFrame with OHLCV data ('close' column required)
            signals: Series with trading signals (1=BUY, 0=HOLD, -1=SELL)
            transaction_cost: Cost per transaction (default 0.1%)
            
        Returns:
            Dictionary with backtesting results
        """
        logger.info("🚀 Starting backtest...")
        
        # Combine data and signals
        df = data.copy()
        df['signal'] = signals
        
        # Forward fill signals to align with price data
        df['signal'] = df['signal'].ffill()
        
        # Calculate position (1 for long, 0 for no position, -1 for short)
        df['position'] = df['signal'].shift(1).fillna(0)  # Use previous day's signal
        
        # Calculate daily returns
        df['returns'] = df['close'].pct_change().fillna(0)
        
        # Calculate strategy returns
        df['strategy_returns'] = df['position'] * df['returns']
        
        # Account for transaction costs
        df['position_changed'] = df['position'].diff().fillna(0) != 0
        df['transaction_costs'] = df['position_changed'].astype(int) * transaction_cost
        df['strategy_returns'] = df['strategy_returns'] - df['transaction_costs']
        
        # Calculate equity curve
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        df['equity_curve'] = self.initial_capital * df['cumulative_returns']
        
        # Calculate benchmark returns (buy and hold)
        df['benchmark_returns'] = (1 + df['returns']).cumprod()
        df['benchmark_equity'] = self.initial_capital * df['benchmark_returns']
        
        # Calculate drawdowns
        df['rolling_max'] = df['equity_curve'].expanding().max()
        df['drawdown'] = (df['equity_curve'] - df['rolling_max']) / df['rolling_max']
        df['benchmark_rolling_max'] = df['benchmark_equity'].expanding().max()
        df['benchmark_drawdown'] = (df['benchmark_equity'] - df['benchmark_rolling_max']) / df['benchmark_rolling_max']
        
        # Calculate backtest metrics
        results = self._calculate_metrics(df)
        
        self.results = results
        logger.info(f"✅ Backtest completed. Final capital: ${results['final_capital']:.2f}")
        
        return results
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics from backtest results
        
        Args:
            df: DataFrame with backtest results
            
        Returns:
            Dictionary with performance metrics
        """
        # Strategy metrics
        strategy_returns = df['strategy_returns'].dropna()
        benchmark_returns = df['returns'].dropna()
        
        # Total return
        total_return = df['equity_curve'].iloc[-1] / self.initial_capital - 1
        benchmark_total_return = df['benchmark_equity'].iloc[-1] / self.initial_capital - 1
        
        # Annualized return (assuming 252 trading days)
        years = len(df) / 252
        annualized_return = (df['equity_curve'].iloc[-1] / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0
        benchmark_annualized_return = (df['benchmark_equity'].iloc[-1] / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        strategy_volatility = strategy_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / strategy_volatility if strategy_volatility > 0 else 0
        benchmark_sharpe_ratio = benchmark_annualized_return / benchmark_volatility if benchmark_volatility > 0 else 0
        
        # Max drawdown
        max_drawdown = df['drawdown'].min()
        benchmark_max_drawdown = df['benchmark_drawdown'].min()
        
        # Win rate
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if len(winning_trades) + len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Number of trades
        num_trades = len(df[df['position_changed']])
        
        # Calmar ratio (return over max drawdown)
        calmar_ratio = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': df['equity_curve'].iloc[-1],
            'total_return': total_return,
            'benchmark_total_return': benchmark_total_return,
            'annualized_return': annualized_return,
            'benchmark_annualized_return': benchmark_annualized_return,
            'strategy_volatility': strategy_volatility,
            'benchmark_volatility': benchmark_volatility,
            'sharpe_ratio': sharpe_ratio,
            'benchmark_sharpe_ratio': benchmark_sharpe_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades,
            'best_trade': winning_trades.max() if len(winning_trades) > 0 else 0,
            'worst_trade': losing_trades.min() if len(losing_trades) > 0 else 0,
            'avg_win': gross_profit / len(winning_trades) if len(winning_trades) > 0 else 0,
            'avg_loss': gross_loss / len(losing_trades) if len(losing_trades) > 0 else 0
        }
        
        return metrics
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.results:
            logger.error("No backtest results to plot. Run backtest first.")
            return
        
        # We'll create a placeholder plot since we don't have the actual data
        plt.figure(figsize=(12, 8))
        
        # Plot title
        plt.suptitle('Backtest Results', fontsize=16)
        
        # Equity curves subplot
        plt.subplot(2, 2, 1)
        plt.title('Equity Curves')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        # Placeholder - in real implementation this would plot actual equity curves
        
        # Drawdown subplot
        plt.subplot(2, 2, 2)
        plt.title('Drawdowns')
        plt.xlabel('Time')
        plt.ylabel('Drawdown')
        # Placeholder - in real implementation this would plot actual drawdowns
        
        # Monthly returns subplot
        plt.subplot(2, 2, 3)
        plt.title('Monthly Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        # Placeholder - in real implementation this would plot actual returns distribution
        
        # Performance metrics subplot
        plt.subplot(2, 2, 4)
        plt.title('Performance Metrics')
        plt.axis('off')
        
        # Display key metrics
        metrics_text = f"""
        Total Return: {self.results['total_return']:.2%}
        Annualized Return: {self.results['annualized_return']:.2%}
        Sharpe Ratio: {self.results['sharpe_ratio']:.2f}
        Max Drawdown: {self.results['max_drawdown']:.2%}
        Win Rate: {self.results['win_rate']:.2%}
        Profit Factor: {self.results['profit_factor']:.2f}
        """
        plt.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"📈 Backtest results saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a textual report of backtest results
        
        Returns:
            Formatted string with backtest report
        """
        if not self.results:
            return "No backtest results available. Run backtest first."
        
        report = f"""
BACKTEST RESULTS REPORT
=======================

PERFORMANCE METRICS:
- Initial Capital: ${self.results['initial_capital']:,.2f}
- Final Capital: ${self.results['final_capital']:,.2f}
- Total Return: {self.results['total_return']:.2%}
- Benchmark Total Return: {self.results['benchmark_total_return']:.2%}
- Annualized Return: {self.results['annualized_return']:.2%}
- Benchmark Annualized Return: {self.results['benchmark_annualized_return']:.2%}

RISK METRICS:
- Strategy Volatility: {self.results['strategy_volatility']:.2%}
- Benchmark Volatility: {self.results['benchmark_volatility']:.2%}
- Sharpe Ratio: {self.results['sharpe_ratio']:.2f}
- Benchmark Sharpe Ratio: {self.results['benchmark_sharpe_ratio']:.2f}
- Max Drawdown: {self.results['max_drawdown']:.2%}
- Benchmark Max Drawdown: {self.results['benchmark_max_drawdown']:.2%}
- Calmar Ratio: {self.results['calmar_ratio']:.2f}

TRADING METRICS:
- Win Rate: {self.results['win_rate']:.2%}
- Profit Factor: {self.results['profit_factor']:.2f}
- Number of Trades: {self.results['num_trades']}
- Best Trade: {self.results['best_trade']:.2%}
- Worst Trade: {self.results['worst_trade']:.2%}
- Avg Win: {self.results['avg_win']:.2%}
- Avg Loss: {self.results['avg_loss']:.2%}

SUMMARY:
- The strategy {'outperformed' if self.results['total_return'] > self.results['benchmark_total_return'] else 'underperformed'} the benchmark
- {'Positive' if self.results['total_return'] > 0 else 'Negative'} returns achieved
- {'Acceptable' if abs(self.results['max_drawdown']) < 0.15 else 'High'} drawdown levels
- {'Good' if self.results['win_rate'] > 0.5 else 'Poor'} win rate
        """
        
        return report

class WalkForwardAnalyzer:
    """
    Performs walk-forward analysis to validate strategy robustness
    """
    
    def __init__(self, in_sample_window: int = 252, out_of_sample_window: int = 63):
        self.in_sample_window = in_sample_window  # 1 year for training
        self.out_of_sample_window = out_of_sample_window  # 3 months for testing
    
    def run_walk_forward_analysis(self, data: pd.DataFrame, signals_generator_func, 
                                metric_to_optimize: str = 'sharpe_ratio') -> Dict:
        """
        Run walk-forward analysis on the data
        
        Args:
            data: DataFrame with historical data
            signals_generator_func: Function that generates signals based on data slice
            metric_to_optimize: Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')
            
        Returns:
            Dictionary with walk-forward analysis results
        """
        logger.info("🔄 Running walk-forward analysis...")
        
        results = []
        
        # Start with in-sample period
        start_idx = 0
        while start_idx + self.in_sample_window + self.out_of_sample_window <= len(data):
            # Define in-sample and out-of-sample periods
            in_sample_end = start_idx + self.in_sample_window
            oos_start = in_sample_end
            oos_end = oos_start + self.out_of_sample_window
            
            # Get data slices
            in_sample_data = data.iloc[start_idx:in_sample_end]
            oos_data = data.iloc[oos_start:oos_end]
            
            # Generate signals on in-sample data (this would typically involve retraining)
            # For simplicity, we'll use the same signal generator
            in_sample_signals = signals_generator_func(in_sample_data)
            oos_signals = signals_generator_func(oos_data)
            
            # Run backtest on out-of-sample data
            backtester = Backtester()
            oos_results = backtester.run_backtest(oos_data, oos_signals)
            
            results.append({
                'period_start': data.index[oos_start],
                'period_end': data.index[oos_end],
                'results': oos_results
            })
            
            # Move to next period
            start_idx = oos_end
        
        # Aggregate results
        aggregate_results = self._aggregate_walk_forward_results(results)
        
        logger.info("✅ Walk-forward analysis completed")
        return aggregate_results
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate walk-forward analysis results
        
        Args:
            results: List of period results
            
        Returns:
            Aggregated results dictionary
        """
        if not results:
            return {}
        
        # Extract metrics for aggregation
        total_returns = [r['results']['total_return'] for r in results]
        sharpe_ratios = [r['results']['sharpe_ratio'] for r in results]
        max_drawdowns = [r['results']['max_drawdown'] for r in results]
        win_rates = [r['results']['win_rate'] for r in results]
        
        aggregate = {
            'num_periods': len(results),
            'avg_total_return': np.mean(total_returns),
            'std_total_return': np.std(total_returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'std_sharpe_ratio': np.std(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'std_max_drawdown': np.std(max_drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'consistency_score': self._calculate_consistency_score(results)
        }
        
        return aggregate
    
    def _calculate_consistency_score(self, results: List[Dict]) -> float:
        """
        Calculate consistency score based on period-to-period performance stability
        
        Args:
            results: List of period results
            
        Returns:
            Consistency score (0-1, higher is better)
        """
        if len(results) < 2:
            return 1.0
        
        # Calculate coefficient of variation for key metrics
        total_returns = [r['results']['total_return'] for r in results]
        sharpe_ratios = [r['results']['sharpe_ratio'] for r in results]
        
        # Coefficient of variation (lower is more consistent)
        cv_returns = np.std(total_returns) / np.abs(np.mean(total_returns)) if np.mean(total_returns) != 0 else float('inf')
        cv_sharpe = np.std(sharpe_ratios) / np.abs(np.mean(sharpe_ratios)) if np.mean(sharpe_ratios) != 0 else float('inf')
        
        # Convert to consistency score (inverse relationship)
        consistency_score = 1 / (1 + cv_returns + cv_sharpe)
        
        return min(consistency_score, 1.0)  # Cap at 1.0

# Example usage
def main():
    """Example usage of the Backtester module"""
    print("Backtesting module ready for strategy evaluation")

if __name__ == "__main__":
    main()