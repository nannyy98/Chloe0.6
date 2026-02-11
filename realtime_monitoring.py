"""
Real-time Monitoring Dashboard for Chloe AI 0.4
Professional monitoring system with live metrics, alerts, and performance tracking
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import json
import numpy as np
import pandas as pd
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Real-time system performance metrics"""
    timestamp: datetime
    portfolio_value: float
    cash_balance: float
    positions_value: float
    total_pnl: float
    total_return_pct: float
    current_drawdown: float
    number_of_positions: int
    active_orders: int
    daily_pnl: float
    daily_return_pct: float
    volatility_30d: float
    sharpe_ratio: float
    win_rate: float

@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    timestamp: datetime
    portfolio_exposure: float
    max_position_size: float
    correlation_risk: float
    liquidity_risk: float
    regime_state: str
    regime_confidence: float
    var_95: float
    cvar_95: float
    stress_test_results: Dict[str, float]

@dataclass
class Alert:
    """Monitoring alert"""
    alert_id: str
    timestamp: datetime
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    category: str  # 'RISK', 'PERFORMANCE', 'SYSTEM'
    message: str
    triggered_value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class MetricsCollector:
    """Collects and calculates real-time metrics"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 data points
        self.risk_history = deque(maxlen=1000)
        logger.info("📊 Metrics Collector initialized")

    def collect_system_metrics(self, portfolio_data: Dict, risk_data: Dict) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        try:
            timestamp = datetime.now()
            
            # Portfolio metrics
            portfolio_value = portfolio_data.get('total_value', 0)
            cash_balance = portfolio_data.get('cash_balance', 0)
            positions_value = portfolio_data.get('positions_value', 0)
            total_pnl = portfolio_data.get('total_pnl', 0)
            
            # Calculate returns
            initial_capital = portfolio_data.get('initial_capital', 100000)
            total_return_pct = ((portfolio_value - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0
            
            # Drawdown calculation
            peak_value = max([m.portfolio_value for m in self.metrics_history] + [portfolio_value])
            current_drawdown = ((peak_value - portfolio_value) / peak_value) * 100 if peak_value > 0 else 0
            
            # Daily metrics
            daily_pnl = self._calculate_daily_pnl()
            daily_return_pct = (daily_pnl / portfolio_value * 100) if portfolio_value > 0 else 0
            
            # Risk-adjusted metrics
            volatility_30d = self._calculate_volatility_30d()
            sharpe_ratio = self._calculate_sharpe_ratio(volatility_30d)
            win_rate = self._calculate_win_rate()
            
            metrics = SystemMetrics(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                cash_balance=cash_balance,
                positions_value=positions_value,
                total_pnl=total_pnl,
                total_return_pct=total_return_pct,
                current_drawdown=current_drawdown,
                number_of_positions=len(portfolio_data.get('positions', [])),
                active_orders=len(portfolio_data.get('active_orders', [])),
                daily_pnl=daily_pnl,
                daily_return_pct=daily_return_pct,
                volatility_30d=volatility_30d,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate
            )
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return self._get_default_metrics()

    def collect_risk_metrics(self, risk_data: Dict, market_data: Dict) -> RiskMetrics:
        """Collect risk-related metrics"""
        try:
            timestamp = datetime.now()
            
            risk_metrics = RiskMetrics(
                timestamp=timestamp,
                portfolio_exposure=risk_data.get('portfolio_exposure', 0),
                max_position_size=risk_data.get('max_position_size', 0),
                correlation_risk=risk_data.get('correlation_risk', 0),
                liquidity_risk=risk_data.get('liquidity_risk', 0),
                regime_state=risk_data.get('regime_state', 'STABLE'),
                regime_confidence=risk_data.get('regime_confidence', 0.5),
                var_95=risk_data.get('var_95', 0),
                cvar_95=risk_data.get('cvar_95', 0),
                stress_test_results=risk_data.get('stress_test_results', {})
            )
            
            self.risk_history.append(risk_metrics)
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Risk metrics collection failed: {e}")
            return self._get_default_risk_metrics()

    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        today = datetime.now().date()
        today_metrics = [m for m in self.metrics_history if m.timestamp.date() == today]
        
        if len(today_metrics) < 2:
            return 0.0
            
        return today_metrics[-1].total_pnl - today_metrics[0].total_pnl

    def _calculate_volatility_30d(self) -> float:
        """Calculate 30-day volatility"""
        if len(self.metrics_history) < 30:
            return 0.0
            
        recent_returns = []
        recent_metrics = list(self.metrics_history)[-30:]
        
        for i in range(1, len(recent_metrics)):
            prev_value = recent_metrics[i-1].portfolio_value
            curr_value = recent_metrics[i].portfolio_value
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                recent_returns.append(daily_return)
        
        if not recent_returns:
            return 0.0
            
        return np.std(recent_returns) * np.sqrt(252) * 100  # Annualized percentage

    def _calculate_sharpe_ratio(self, volatility: float) -> float:
        """Calculate Sharpe ratio"""
        if len(self.metrics_history) < 30 or volatility <= 0:
            return 0.0
            
        recent_metrics = list(self.metrics_history)[-30:]
        returns = []
        
        for i in range(1, len(recent_metrics)):
            prev_value = recent_metrics[i-1].portfolio_value
            curr_value = recent_metrics[i].portfolio_value
            if prev_value > 0:
                ret = (curr_value - prev_value) / prev_value
                returns.append(ret)
        
        if not returns:
            return 0.0
            
        avg_return = np.mean(returns)
        risk_free_rate = 0.02 / 252  # 2% annual risk-free rate
        excess_return = avg_return - risk_free_rate
        
        return (excess_return / (np.std(returns) or 1)) * np.sqrt(252)

    def _calculate_win_rate(self) -> float:
        """Calculate trade win rate"""
        if len(self.metrics_history) < 10:
            return 0.0
            
        # Simplified win rate calculation based on positive/negative returns
        positive_periods = 0
        total_periods = 0
        
        recent_metrics = list(self.metrics_history)[-30:]
        for i in range(1, len(recent_metrics)):
            prev_pnl = recent_metrics[i-1].total_pnl
            curr_pnl = recent_metrics[i].total_pnl
            if curr_pnl != prev_pnl:
                total_periods += 1
                if curr_pnl > prev_pnl:
                    positive_periods += 1
        
        return (positive_periods / total_periods * 100) if total_periods > 0 else 0.0

    def _get_default_metrics(self) -> SystemMetrics:
        """Return default metrics when calculation fails"""
        return SystemMetrics(
            timestamp=datetime.now(),
            portfolio_value=100000,
            cash_balance=100000,
            positions_value=0,
            total_pnl=0,
            total_return_pct=0,
            current_drawdown=0,
            number_of_positions=0,
            active_orders=0,
            daily_pnl=0,
            daily_return_pct=0,
            volatility_30d=0,
            sharpe_ratio=0,
            win_rate=0
        )

    def _get_default_risk_metrics(self) -> RiskMetrics:
        """Return default risk metrics when calculation fails"""
        return RiskMetrics(
            timestamp=datetime.now(),
            portfolio_exposure=0,
            max_position_size=0,
            correlation_risk=0,
            liquidity_risk=0,
            regime_state='STABLE',
            regime_confidence=0.5,
            var_95=0,
            cvar_95=0,
            stress_test_results={}
        )

class AlertManager:
    """Manages real-time alerts and notifications"""
    
    def __init__(self):
        self.alerts = []
        self.active_alerts = []
        self.alert_rules = self._initialize_alert_rules()
        logger.info("🔔 Alert Manager initialized")

    def _initialize_alert_rules(self) -> Dict:
        """Initialize alert threshold rules"""
        return {
            'drawdown_critical': {'threshold': 15.0, 'severity': 'CRITICAL', 'type': 'RISK'},
            'drawdown_warning': {'threshold': 8.0, 'severity': 'WARNING', 'type': 'RISK'},
            'exposure_critical': {'threshold': 50.0, 'severity': 'CRITICAL', 'type': 'RISK'},
            'exposure_warning': {'threshold': 30.0, 'severity': 'WARNING', 'type': 'RISK'},
            'correlation_high': {'threshold': 0.8, 'severity': 'WARNING', 'type': 'RISK'},
            'daily_loss_critical': {'threshold': -3.0, 'severity': 'CRITICAL', 'type': 'PERFORMANCE'},
            'daily_loss_warning': {'threshold': -1.5, 'severity': 'WARNING', 'type': 'PERFORMANCE'},
            'system_error': {'threshold': 1, 'severity': 'CRITICAL', 'type': 'SYSTEM'}
        }

    def check_alerts(self, system_metrics: SystemMetrics, risk_metrics: RiskMetrics) -> List[Alert]:
        """Check for triggered alerts"""
        new_alerts = []
        
        # Drawdown alerts
        if system_metrics.current_drawdown >= self.alert_rules['drawdown_critical']['threshold']:
            new_alerts.append(self._create_alert(
                'drawdown_critical', system_metrics.current_drawdown,
                self.alert_rules['drawdown_critical']['threshold'],
                f"_CRITICAL DRAWDOWN_: {system_metrics.current_drawdown:.2f}% exceeds {self.alert_rules['drawdown_critical']['threshold']:.1f}% threshold"
            ))
        elif system_metrics.current_drawdown >= self.alert_rules['drawdown_warning']['threshold']:
            new_alerts.append(self._create_alert(
                'drawdown_warning', system_metrics.current_drawdown,
                self.alert_rules['drawdown_warning']['threshold'],
                f"_High drawdown_: {system_metrics.current_drawdown:.2f}% (warning at {self.alert_rules['drawdown_warning']['threshold']:.1f}%)"
            ))
        
        # Exposure alerts
        if risk_metrics.portfolio_exposure >= self.alert_rules['exposure_critical']['threshold']:
            new_alerts.append(self._create_alert(
                'exposure_critical', risk_metrics.portfolio_exposure,
                self.alert_rules['exposure_critical']['threshold'],
                f"_CRITICAL EXPOSURE_: {risk_metrics.portfolio_exposure:.1f}% exceeds {self.alert_rules['exposure_critical']['threshold']:.1f}% threshold"
            ))
        elif risk_metrics.portfolio_exposure >= self.alert_rules['exposure_warning']['threshold']:
            new_alerts.append(self._create_alert(
                'exposure_warning', risk_metrics.portfolio_exposure,
                self.alert_rules['exposure_warning']['threshold'],
                f"_High exposure_: {risk_metrics.portfolio_exposure:.1f}% (warning at {self.alert_rules['exposure_warning']['threshold']:.1f}%)"
            ))
        
        # Daily performance alerts
        if system_metrics.daily_return_pct <= self.alert_rules['daily_loss_critical']['threshold']:
            new_alerts.append(self._create_alert(
                'daily_loss_critical', system_metrics.daily_return_pct,
                self.alert_rules['daily_loss_critical']['threshold'],
                f"_CRITICAL DAILY LOSS_: {system_metrics.daily_return_pct:.2f}% exceeds {self.alert_rules['daily_loss_critical']['threshold']:.1f}% threshold"
            ))
        elif system_metrics.daily_return_pct <= self.alert_rules['daily_loss_warning']['threshold']:
            new_alerts.append(self._create_alert(
                'daily_loss_warning', system_metrics.daily_return_pct,
                self.alert_rules['daily_loss_warning']['threshold'],
                f"_Daily loss warning_: {system_metrics.daily_return_pct:.2f}% (warning at {self.alert_rules['daily_loss_warning']['threshold']:.1f}%)"
            ))
        
        # Add new alerts
        for alert in new_alerts:
            if not self._alert_already_exists(alert):
                self.alerts.append(alert)
                self.active_alerts.append(alert)
                logger.warning(f"🚨 ALERT TRIGGERED: {alert.message}")
        
        return new_alerts

    def _create_alert(self, rule_name: str, value: float, threshold: float, message: str) -> Alert:
        """Create new alert instance"""
        return Alert(
            alert_id=f"{rule_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            severity=self.alert_rules[rule_name]['severity'],
            category=self.alert_rules[rule_name]['type'],
            message=message,
            triggered_value=value,
            threshold=threshold
        )

    def _alert_already_exists(self, new_alert: Alert) -> bool:
        """Check if similar alert already exists"""
        for existing_alert in self.active_alerts:
            if (existing_alert.category == new_alert.category and 
                existing_alert.severity == new_alert.severity and
                not existing_alert.resolved):
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved"""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"✅ Alert resolved: {alert_id}")
                return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts"""
        return [alert for alert in self.active_alerts if not alert.resolved]

class RealTimeMonitor:
    """Main real-time monitoring system"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.is_monitoring = False
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
        # Data storage
        self.system_metrics_history = deque(maxlen=1000)
        self.risk_metrics_history = deque(maxlen=1000)
        self.recent_alerts = deque(maxlen=100)
        
        logger.info("🖥️ Real-time Monitor initialized")
        logger.info(f"   Update interval: {update_interval}s")

    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.is_monitoring = True
        logger.info("🟢 Real-time monitoring started")
        
        while self.is_monitoring:
            try:
                await self._monitoring_cycle()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Monitoring cycle failed: {e}")
                await asyncio.sleep(5)  # Wait before retry

    async def _monitoring_cycle(self):
        """Single monitoring cycle"""
        # In real system, this would fetch actual data from components
        portfolio_data = self._get_mock_portfolio_data()
        risk_data = self._get_mock_risk_data()
        market_data = self._get_mock_market_data()
        
        # Collect metrics
        system_metrics = self.metrics_collector.collect_system_metrics(portfolio_data, risk_data)
        risk_metrics = self.metrics_collector.collect_risk_metrics(risk_data, market_data)
        
        # Store metrics
        self.system_metrics_history.append(system_metrics)
        self.risk_metrics_history.append(risk_metrics)
        
        # Check for alerts
        new_alerts = self.alert_manager.check_alerts(system_metrics, risk_metrics)
        self.recent_alerts.extend(new_alerts)
        
        # Log periodic summary
        if len(self.system_metrics_history) % 10 == 0:  # Every 10 cycles
            self._log_monitoring_summary(system_metrics, risk_metrics)

    def _get_mock_portfolio_data(self) -> Dict:
        """Mock portfolio data (would be real data in production)"""
        # Simulate realistic portfolio movements
        base_value = 100000
        noise = np.random.normal(0, 0.001)  # Small random fluctuations
        trend = 0.0001  # Small upward trend
        
        current_value = base_value * (1 + trend + noise)
        
        return {
            'total_value': current_value,
            'cash_balance': current_value * 0.2,  # 20% cash
            'positions_value': current_value * 0.8,  # 80% positions
            'total_pnl': current_value - base_value,
            'initial_capital': base_value,
            'positions': ['BTC/USDT', 'ETH/USDT'],
            'active_orders': []
        }

    def _get_mock_risk_data(self) -> Dict:
        """Mock risk data (would be real data in production)"""
        return {
            'portfolio_exposure': np.random.uniform(15, 35),
            'max_position_size': np.random.uniform(5, 15),
            'correlation_risk': np.random.uniform(0.3, 0.7),
            'liquidity_risk': np.random.uniform(0.1, 0.4),
            'regime_state': np.random.choice(['STABLE', 'TRENDING', 'VOLATILE']),
            'regime_confidence': np.random.uniform(0.6, 0.9),
            'var_95': np.random.uniform(1000, 3000),
            'cvar_95': np.random.uniform(2000, 5000),
            'stress_test_results': {
                'market_crash': np.random.uniform(-0.15, -0.05),
                'volatility_spike': np.random.uniform(-0.10, -0.02),
                'liquidity_dryup': np.random.uniform(-0.08, -0.01)
            }
        }

    def _get_mock_market_data(self) -> Dict:
        """Mock market data"""
        return {
            'prices': {
                'BTC/USDT': 48500 + np.random.normal(0, 100),
                'ETH/USDT': 3650 + np.random.normal(0, 50),
                'SOL/USDT': 47.5 + np.random.normal(0, 1)
            },
            'volatility': {
                'BTC/USDT': 0.03 + np.random.normal(0, 0.005),
                'ETH/USDT': 0.04 + np.random.normal(0, 0.005),
                'SOL/USDT': 0.06 + np.random.normal(0, 0.01)
            }
        }

    def _log_monitoring_summary(self, system_metrics: SystemMetrics, risk_metrics: RiskMetrics):
        """Log periodic monitoring summary"""
        active_alerts = len(self.alert_manager.get_active_alerts())
        
        logger.info(f"📊 MONITORING SUMMARY:")
        logger.info(f"   Portfolio: ${system_metrics.portfolio_value:,.2f} ({system_metrics.total_return_pct:+.2f}%)")
        logger.info(f"   Drawdown: {system_metrics.current_drawdown:.2f}%")
        logger.info(f"   Exposure: {risk_metrics.portfolio_exposure:.1f}%")
        logger.info(f"   Regime: {risk_metrics.regime_state} ({risk_metrics.regime_confidence:.2f})")
        logger.info(f"   Active Alerts: {active_alerts}")
        logger.info(f"   Sharpe Ratio: {system_metrics.sharpe_ratio:.2f}")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        logger.info("🔴 Real-time monitoring stopped")

    def get_dashboard_data(self) -> Dict:
        """Get current dashboard data for display"""
        latest_system = self.system_metrics_history[-1] if self.system_metrics_history else None
        latest_risk = self.risk_metrics_history[-1] if self.risk_metrics_history else None
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': asdict(latest_system) if latest_system else {},
            'risk_metrics': asdict(latest_risk) if latest_risk else {},
            'active_alerts': [asdict(alert) for alert in active_alerts],
            'monitoring_status': 'RUNNING' if self.is_monitoring else 'STOPPED',
            'system_uptime': len(self.system_metrics_history) * self.update_interval
        }

# Global monitor instance
monitor = None

def get_monitor(update_interval: float = 1.0) -> RealTimeMonitor:
    """Get singleton monitor instance"""
    global monitor
    if monitor is None:
        monitor = RealTimeMonitor(update_interval)
    return monitor

def main():
    """Example usage"""
    print("🖥️ Real-time Monitoring System ready")

if __name__ == "__main__":
    main()