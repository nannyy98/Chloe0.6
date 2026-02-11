"""
Advanced Risk Models for Chloe AI
Implementation of VaR, Monte Carlo simulations, and stress testing
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_normal: float
    var_historical: float
    var_monte_carlo: float
    expected_shortfall: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    value_at_risk_conf: float
    time_horizon: str
    confidence_level: float

@dataclass
class StressTestResult:
    """Stress test results"""
    scenario_name: str
    portfolio_value: float
    portfolio_loss: float
    loss_percentage: float
    risk_metrics: RiskMetrics
    timestamp: datetime

@dataclass
class MonteCarloSimulation:
    """Monte Carlo simulation results"""
    simulated_returns: np.ndarray
    portfolio_values: np.ndarray
    var_estimate: float
    expected_shortfall: float
    confidence_intervals: Dict[str, float]
    scenario_count: int

class AdvancedRiskAnalyzer:
    """
    Advanced risk analysis with multiple methodologies
    """
    
    def __init__(self, confidence_level: float = 0.95, time_horizon: int = 1):
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.percentile = (1 - confidence_level) * 100
        
        # Stress test scenarios
        self.stress_scenarios = {
            'market_crash_2008': {
                'description': '2008 Financial Crisis scenario',
                'shock_multiplier': -0.30,  # 30% market drop
                'volatility_multiplier': 3.0,  # 3x volatility
                'correlation_increase': 0.2  # Increased correlation
            },
            'crypto_crash_2022': {
                'description': '2022 Crypto Market Crash',
                'shock_multiplier': -0.60,  # 60% crypto drop
                'volatility_multiplier': 4.0,
                'liquidity_shock': True
            },
            'black_swan': {
                'description': 'Black Swan event',
                'shock_multiplier': -0.50,
                'volatility_multiplier': 5.0,
                'correlation_to_1': True
            },
            'interest_rate_spike': {
                'description': 'Sudden interest rate increase',
                'shock_multiplier': -0.15,
                'volatility_multiplier': 2.0,
                'duration_shock': 30  # days
            },
            'regulatory_crackdown': {
                'description': 'Severe regulatory intervention',
                'shock_multiplier': -0.40,
                'volatility_multiplier': 3.5,
                'sector_specific': True
            }
        }
        
        logger.info("🛡️ Advanced Risk Analyzer initialized")
    
    def calculate_value_at_risk(self, returns: np.ndarray, 
                              portfolio_value: float = 10000.0,
                              method: str = 'all') -> Dict[str, float]:
        """
        Calculate Value at Risk using multiple methods
        
        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value
            method: 'normal', 'historical', 'monte_carlo', or 'all'
            
        Returns:
            Dictionary with VaR calculations
        """
        var_results = {}
        
        if method in ['normal', 'all']:
            var_results['normal'] = self._calculate_var_normal(returns, portfolio_value)
        
        if method in ['historical', 'all']:
            var_results['historical'] = self._calculate_var_historical(returns, portfolio_value)
        
        if method in ['monte_carlo', 'all']:
            var_results['monte_carlo'] = self._calculate_var_monte_carlo(returns, portfolio_value)
        
        return var_results
    
    def _calculate_var_normal(self, returns: np.ndarray, portfolio_value: float) -> float:
        """Calculate VaR using normal distribution method"""
        if len(returns) < 30:
            logger.warning("⚠️ Insufficient data for normal VaR calculation")
            return 0.0
        
        # Calculate mean and standard deviation
        mean_return = np.mean(returns)
        std_dev = np.std(returns)
        
        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - self.confidence_level)
        var_normal = portfolio_value * (mean_return - z_score * std_dev * np.sqrt(self.time_horizon))
        
        return max(0, var_normal)  # VaR should be positive (loss)
    
    def _calculate_var_historical(self, returns: np.ndarray, portfolio_value: float) -> float:
        """Calculate VaR using historical simulation method"""
        if len(returns) < 30:
            logger.warning("⚠️ Insufficient data for historical VaR calculation")
            return 0.0
        
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Find percentile
        percentile_index = int(np.ceil(len(sorted_returns) * (1 - self.confidence_level))) - 1
        percentile_index = max(0, min(percentile_index, len(sorted_returns) - 1))
        
        # Calculate VaR
        var_historical = -portfolio_value * sorted_returns[percentile_index] * np.sqrt(self.time_horizon)
        
        return max(0, var_historical)
    
    def _calculate_var_monte_carlo(self, returns: np.ndarray, portfolio_value: float, 
                                 simulations: int = 10000) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        if len(returns) < 30:
            logger.warning("⚠️ Insufficient data for Monte Carlo VaR calculation")
            return 0.0
        
        # Fit distribution to returns
        mean_return = np.mean(returns)
        std_dev = np.std(returns)
        
        # Generate simulated returns
        simulated_returns = np.random.normal(mean_return, std_dev, simulations)
        
        # Calculate portfolio values
        simulated_portfolio_values = portfolio_value * (1 + simulated_returns)
        losses = portfolio_value - simulated_portfolio_values
        
        # Calculate VaR
        var_monte_carlo = np.percentile(losses, self.percentile)
        
        return max(0, var_monte_carlo)
    
    def calculate_expected_shortfall(self, returns: np.ndarray, portfolio_value: float) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value
            
        Returns:
            Expected Shortfall value
        """
        if len(returns) < 50:
            logger.warning("⚠️ Insufficient data for Expected Shortfall calculation")
            return 0.0
        
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Find VaR threshold
        var_threshold = np.percentile(sorted_returns, self.percentile)
        
        # Calculate average of returns below VaR threshold
        tail_returns = sorted_returns[sorted_returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return 0.0
        
        expected_shortfall = -portfolio_value * np.mean(tail_returns) * np.sqrt(self.time_horizon)
        
        return max(0, expected_shortfall)
    
    def run_monte_carlo_simulation(self, initial_portfolio: float,
                                 assets_returns: Dict[str, np.ndarray],
                                 weights: Dict[str, float],
                                 simulations: int = 10000,
                                 time_periods: int = 252) -> MonteCarloSimulation:
        """
        Run Monte Carlo simulation for portfolio risk assessment
        
        Args:
            initial_portfolio: Initial portfolio value
            assets_returns: Dictionary of asset returns arrays
            weights: Portfolio weights for each asset
            simulations: Number of simulations
            time_periods: Number of time periods to simulate
            
        Returns:
            MonteCarloSimulation object with results
        """
        logger.info(f"🎲 Running Monte Carlo simulation ({simulations} simulations)")
        
        # Validate inputs
        if not assets_returns or not weights:
            raise ValueError("Assets returns and weights must be provided")
        
        # Align assets and weights
        asset_names = list(weights.keys())
        portfolio_weights = np.array([weights[asset] for asset in asset_names])
        
        # Calculate portfolio statistics
        portfolio_returns = []
        for asset in asset_names:
            if asset in assets_returns and len(assets_returns[asset]) > 0:
                portfolio_returns.append(assets_returns[asset])
        
        if not portfolio_returns:
            raise ValueError("No valid return data found")
        
        # Calculate portfolio mean and covariance
        returns_matrix = np.column_stack(portfolio_returns)
        portfolio_mean = np.dot(portfolio_weights, np.mean(returns_matrix, axis=0))
        portfolio_cov = np.cov(returns_matrix.T)
        
        # Generate correlated random returns
        simulated_portfolio_returns = np.zeros((simulations, time_periods))
        
        for i in range(simulations):
            # Generate correlated random returns for each time period
            period_returns = np.random.multivariate_normal(
                np.mean(returns_matrix, axis=0),
                portfolio_cov,
                time_periods
            )
            
            # Calculate weighted portfolio returns
            weighted_returns = np.dot(period_returns, portfolio_weights)
            simulated_portfolio_returns[i] = weighted_returns
        
        # Calculate cumulative portfolio values
        simulated_portfolio_values = np.zeros((simulations, time_periods + 1))
        simulated_portfolio_values[:, 0] = initial_portfolio
        
        for t in range(1, time_periods + 1):
            simulated_portfolio_values[:, t] = (
                simulated_portfolio_values[:, t-1] * (1 + simulated_portfolio_returns[:, t-1])
            )
        
        # Calculate VaR from simulations
        final_values = simulated_portfolio_values[:, -1]
        losses = initial_portfolio - final_values
        var_estimate = np.percentile(losses, self.percentile)
        
        # Calculate Expected Shortfall
        tail_losses = losses[losses >= var_estimate]
        expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else 0
        
        # Calculate confidence intervals
        confidence_intervals = {
            '5th_percentile': np.percentile(final_values, 5),
            '25th_percentile': np.percentile(final_values, 25),
            'median': np.percentile(final_values, 50),
            '75th_percentile': np.percentile(final_values, 75),
            '95th_percentile': np.percentile(final_values, 95)
        }
        
        result = MonteCarloSimulation(
            simulated_returns=simulated_portfolio_returns,
            portfolio_values=simulated_portfolio_values,
            var_estimate=var_estimate,
            expected_shortfall=expected_shortfall,
            confidence_intervals=confidence_intervals,
            scenario_count=simulations
        )
        
        logger.info(f"✅ Monte Carlo simulation completed")
        logger.info(f"   VaR Estimate: ${var_estimate:,.2f}")
        logger.info(f"   Expected Shortfall: ${expected_shortfall:,.2f}")
        logger.info(f"   Median Final Value: ${confidence_intervals['median']:,.2f}")
        
        return result
    
    def run_stress_tests(self, portfolio_value: float,
                        assets_returns: Dict[str, np.ndarray],
                        weights: Dict[str, float]) -> List[StressTestResult]:
        """
        Run comprehensive stress tests on portfolio
        
        Args:
            portfolio_value: Current portfolio value
            assets_returns: Dictionary of asset returns
            weights: Portfolio weights
            
        Returns:
            List of StressTestResult objects
        """
        logger.info("🚨 Running comprehensive stress tests...")
        
        results = []
        
        for scenario_name, scenario_params in self.stress_scenarios.items():
            try:
                result = self._run_single_stress_test(
                    scenario_name, scenario_params, 
                    portfolio_value, assets_returns, weights
                )
                results.append(result)
                logger.info(f"✅ Completed stress test: {scenario_name}")
                
            except Exception as e:
                logger.error(f"❌ Stress test {scenario_name} failed: {e}")
        
        return results
    
    def _run_single_stress_test(self, scenario_name: str, 
                              scenario_params: Dict,
                              portfolio_value: float,
                              assets_returns: Dict[str, np.ndarray],
                              weights: Dict[str, float]) -> StressTestResult:
        """Run a single stress test scenario"""
        
        # Apply stress scenario to returns
        stressed_returns = {}
        for asset, returns in assets_returns.items():
            if len(returns) > 0:
                # Apply shock multiplier
                shocked_returns = returns * (1 + scenario_params['shock_multiplier'])
                
                # Apply volatility multiplier
                volatility_factor = scenario_params['volatility_multiplier']
                shocked_returns = shocked_returns * volatility_factor
                
                stressed_returns[asset] = shocked_returns
        
        # Calculate portfolio returns under stress
        portfolio_returns = self._calculate_portfolio_returns(stressed_returns, weights)
        
        if len(portfolio_returns) == 0:
            raise ValueError("No valid portfolio returns calculated")
        
        # Calculate stressed portfolio value
        stressed_portfolio_value = portfolio_value * (1 + np.mean(portfolio_returns))
        portfolio_loss = portfolio_value - stressed_portfolio_value
        loss_percentage = (portfolio_loss / portfolio_value) * 100
        
        # Calculate risk metrics under stress
        stress_risk_metrics = self._calculate_stress_risk_metrics(
            portfolio_returns, stressed_portfolio_value
        )
        
        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_value=stressed_portfolio_value,
            portfolio_loss=portfolio_loss,
            loss_percentage=loss_percentage,
            risk_metrics=stress_risk_metrics,
            timestamp=datetime.now()
        )
    
    def _calculate_portfolio_returns(self, assets_returns: Dict[str, np.ndarray],
                                   weights: Dict[str, float]) -> np.ndarray:
        """Calculate weighted portfolio returns"""
        if not assets_returns or not weights:
            return np.array([])
        
        # Align assets and weights
        common_assets = set(assets_returns.keys()) & set(weights.keys())
        if not common_assets:
            return np.array([])
        
        # Get minimum length
        min_length = min(len(assets_returns[asset]) for asset in common_assets if len(assets_returns[asset]) > 0)
        if min_length == 0:
            return np.array([])
        
        # Calculate weighted returns
        portfolio_returns = np.zeros(min_length)
        total_weight = sum(weights[asset] for asset in common_assets)
        
        for asset in common_assets:
            weight = weights[asset] / total_weight if total_weight > 0 else 0
            returns = assets_returns[asset][:min_length]
            portfolio_returns += weight * returns
        
        return portfolio_returns
    
    def _calculate_stress_risk_metrics(self, returns: np.ndarray, 
                                     portfolio_value: float) -> RiskMetrics:
        """Calculate risk metrics for stress scenario"""
        if len(returns) == 0:
            return RiskMetrics(
                var_normal=0, var_historical=0, var_monte_carlo=0,
                expected_shortfall=0, volatility=0, max_drawdown=0,
                sharpe_ratio=0, value_at_risk_conf=0, time_horizon="1d",
                confidence_level=self.confidence_level
            )
        
        # Calculate basic metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Calculate VaR
        var_normal = self._calculate_var_normal(returns, portfolio_value)
        var_historical = self._calculate_var_historical(returns, portfolio_value)
        var_monte_carlo = self._calculate_var_monte_carlo(returns, portfolio_value)
        
        # Calculate Expected Shortfall
        expected_shortfall = self.calculate_expected_shortfall(returns, portfolio_value)
        
        # Calculate Max Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        return RiskMetrics(
            var_normal=var_normal,
            var_historical=var_historical,
            var_monte_carlo=var_monte_carlo,
            expected_shortfall=expected_shortfall,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            value_at_risk_conf=var_normal/portfolio_value if portfolio_value > 0 else 0,
            time_horizon=f"{self.time_horizon}d",
            confidence_level=self.confidence_level
        )
    
    def generate_risk_report(self, portfolio_value: float,
                           assets_returns: Dict[str, np.ndarray],
                           weights: Dict[str, float]) -> Dict:
        """
        Generate comprehensive risk report
        
        Args:
            portfolio_value: Current portfolio value
            assets_returns: Dictionary of asset returns
            weights: Portfolio weights
            
        Returns:
            Comprehensive risk report dictionary
        """
        logger.info("📊 Generating comprehensive risk report...")
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(assets_returns, weights)
        
        if len(portfolio_returns) == 0:
            return {"error": "Insufficient data for risk analysis"}
        
        # Calculate basic risk metrics
        var_results = self.calculate_value_at_risk(portfolio_returns, portfolio_value)
        expected_shortfall = self.calculate_expected_shortfall(portfolio_returns, portfolio_value)
        
        # Run Monte Carlo simulation
        try:
            mc_results = self.run_monte_carlo_simulation(
                portfolio_value, assets_returns, weights
            )
        except Exception as e:
            logger.warning(f"Monte Carlo simulation failed: {e}")
            mc_results = None
        
        # Run stress tests
        try:
            stress_results = self.run_stress_tests(portfolio_value, assets_returns, weights)
        except Exception as e:
            logger.warning(f"Stress tests failed: {e}")
            stress_results = []
        
        # Compile report
        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": portfolio_value,
            "confidence_level": self.confidence_level,
            "time_horizon": f"{self.time_horizon} day(s)",
            "value_at_risk": {
                "normal_method": var_results.get('normal', 0),
                "historical_method": var_results.get('historical', 0),
                "monte_carlo_method": var_results.get('monte_carlo', 0),
                "confidence": f"{self.confidence_level * 100}%"
            },
            "expected_shortfall": expected_shortfall,
            "monte_carlo_analysis": {
                "scenarios_run": mc_results.scenario_count if mc_results else 0,
                "var_estimate": mc_results.var_estimate if mc_results else 0,
                "expected_shortfall": mc_results.expected_shortfall if mc_results else 0,
                "confidence_intervals": mc_results.confidence_intervals if mc_results else {}
            } if mc_results else {},
            "stress_test_results": [
                {
                    "scenario": result.scenario_name,
                    "description": self.stress_scenarios[result.scenario_name]['description'],
                    "portfolio_value": result.portfolio_value,
                    "loss_amount": result.portfolio_loss,
                    "loss_percentage": result.loss_percentage,
                    "volatility": result.risk_metrics.volatility,
                    "max_drawdown": result.risk_metrics.max_drawdown
                }
                for result in stress_results
            ],
            "risk_assessment": self._generate_risk_assessment(var_results, stress_results)
        }
        
        logger.info("✅ Risk report generated successfully")
        return report
    
    def _generate_risk_assessment(self, var_results: Dict, 
                                stress_results: List[StressTestResult]) -> str:
        """Generate human-readable risk assessment"""
        if not var_results:
            return "Insufficient data for risk assessment"
        
        # Get worst VaR estimate
        var_values = [v for v in var_results.values() if v > 0]
        if not var_values:
            return "No valid risk metrics calculated"
        
        worst_var = max(var_values)
        
        # Analyze stress test results
        if stress_results:
            max_loss = max(result.loss_percentage for result in stress_results)
            avg_loss = np.mean([result.loss_percentage for result in stress_results])
        else:
            max_loss = 0
            avg_loss = 0
        
        # Generate assessment
        if worst_var > 0.2 * 10000:  # 20% of portfolio
            risk_level = "HIGH"
        elif worst_var > 0.1 * 10000:  # 10% of portfolio
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        assessment = f"Risk Level: {risk_level}\n"
        assessment += f"Worst Case VaR: ${worst_var:,.2f}\n"
        assessment += f"Maximum Stress Loss: {max_loss:.1f}%\n"
        assessment += f"Average Stress Loss: {avg_loss:.1f}%\n"
        
        if risk_level == "HIGH":
            assessment += "⚠️ High risk detected. Consider reducing exposure."
        elif risk_level == "MEDIUM":
            assessment += "⚠️ Moderate risk level. Monitor positions closely."
        else:
            assessment += "✅ Risk levels are acceptable."
        
        return assessment

# Example usage
def main():
    """Example usage of Advanced Risk Analyzer"""
    analyzer = AdvancedRiskAnalyzer(confidence_level=0.95, time_horizon=1)
    
    # Create sample data
    np.random.seed(42)
    days = 252
    
    # Simulate returns for different assets
    btc_returns = np.random.normal(0.001, 0.04, days)  # 0.1% mean, 4% daily vol
    eth_returns = np.random.normal(0.0005, 0.05, days)  # 0.05% mean, 5% daily vol
    ada_returns = np.random.normal(0.0002, 0.06, days)  # 0.02% mean, 6% daily vol
    
    assets_returns = {
        'BTC': btc_returns,
        'ETH': eth_returns,
        'ADA': ada_returns
    }
    
    weights = {
        'BTC': 0.5,
        'ETH': 0.3,
        'ADA': 0.2
    }
    
    portfolio_value = 10000.0
    
    print("🛡️ Advanced Risk Analysis Demo")
    print("=" * 50)
    
    # Calculate VaR
    var_results = analyzer.calculate_value_at_risk(
        btc_returns, portfolio_value, method='all'
    )
    
    print("Value at Risk (1-day, 95% confidence):")
    for method, var_value in var_results.items():
        print(f"  {method.upper()}: ${var_value:.2f}")
    
    # Run Monte Carlo simulation
    print("\nMonte Carlo Simulation:")
    try:
        mc_results = analyzer.run_monte_carlo_simulation(
            portfolio_value, assets_returns, weights, simulations=1000
        )
        print(f"  VaR Estimate: ${mc_results.var_estimate:.2f}")
        print(f"  Expected Shortfall: ${mc_results.expected_shortfall:.2f}")
        print(f"  Median Final Value: ${mc_results.confidence_intervals['median']:.2f}")
    except Exception as e:
        print(f"  Monte Carlo failed: {e}")
    
    # Run stress tests
    print("\nStress Test Results:")
    try:
        stress_results = analyzer.run_stress_tests(portfolio_value, assets_returns, weights)
        for result in stress_results[:3]:  # Show first 3 scenarios
            print(f"  {result.scenario_name}: {result.loss_percentage:.1f}% loss")
    except Exception as e:
        print(f"  Stress tests failed: {e}")

if __name__ == "__main__":
    main()