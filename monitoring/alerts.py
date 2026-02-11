"""
Professional Monitoring and Alerting System
Real-time monitoring, alerting, and notification for institutional trading
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import pandas as pd

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    RISK_THRESHOLD = "RISK_THRESHOLD"
    DRAWDOWN = "DRAWDOWN"
    POSITION_CONCENTRATION = "POSITION_CONCENTRATION"
    CORRELATION = "CORRELATION"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    VOLUME_ANOMALY = "VOLUME_ANOMALY"
    TECHNICAL_BREAKOUT = "TECHNICAL_BREAKOUT"
    SYSTEM_HEALTH = "SYSTEM_HEALTH"

@dataclass
class Alert:
    """Alert object"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    symbol: Optional[str]
    message: str
    timestamp: datetime
    data: Dict[str, Any]
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

class AlertManager:
    """Professional alert management system"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.active_alerts: List[Alert] = []
        self.alert_callbacks: Dict[AlertType, List[Callable]] = {}
        self.email_config = {}
        self.telegram_config = {}
        self.webhook_config = {}
        
    def register_callback(self, alert_type: AlertType, callback: Callable):
        """Register callback function for alert type"""
        if alert_type not in self.alert_callbacks:
            self.alert_callbacks[alert_type] = []
        self.alert_callbacks[alert_type].append(callback)
        
    def create_alert(self, alert_type: AlertType, severity: AlertSeverity,
                     symbol: Optional[str], message: str, data: Dict[str, Any] = None) -> Alert:
        """Create and trigger alert"""
        alert = Alert(
            alert_id=f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts)}",
            alert_type=alert_type,
            severity=severity,
            symbol=symbol,
            message=message,
            timestamp=datetime.now(),
            data=data or {}
        )
        
        self.alerts.append(alert)
        self.active_alerts.append(alert)
        
        logger.log(
            logging.CRITICAL if severity == AlertSeverity.CRITICAL else
            logging.ERROR if severity == AlertSeverity.HIGH else
            logging.WARNING if severity == AlertSeverity.MEDIUM else
            logging.INFO,
            f"🚨 [{alert.severity.value}] {alert.message}"
        )
        
        # Trigger callbacks
        if alert_type in self.alert_callbacks:
            for callback in self.alert_callbacks[alert_type]:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"❌ Error in alert callback: {e}")
                    
        # Send notifications
        asyncio.create_task(self._send_notifications(alert))
        
        return alert
        
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications via multiple channels"""
        await asyncio.gather(
            self._send_email_notification(alert),
            self._send_telegram_notification(alert),
            self._send_webhook_notification(alert),
            return_exceptions=True
        )
        
    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        if not self.email_config:
            return
            
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"[{alert.severity.value}] {alert.message}"
            msg['From'] = self.email_config.get('from_email')
            msg['To'] = self.email_config.get('to_email')
            
            body = f"""
            ALERT: {alert.message}
            Type: {alert.alert_type.value}
            Severity: {alert.severity.value}
            Symbol: {alert.symbol or 'N/A'}
            Time: {alert.timestamp}
            Data: {json.dumps(alert.data, indent=2, default=str)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config.get('smtp_server'), self.email_config.get('smtp_port'))
            server.starttls()
            server.login(self.email_config.get('username'), self.email_config.get('password'))
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"❌ Failed to send email notification: {e}")
            
    async def _send_telegram_notification(self, alert: Alert):
        """Send Telegram notification"""
        if not self.telegram_config:
            return
            
        try:
            message = f"""
🚨 *ALERT*: {alert.message}
*Type*: {alert.alert_type.value}
*Severity*: {alert.severity.value}
*Symbol*: {alert.symbol or 'N/A'}
*Time*: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            url = f"https://api.telegram.org/bot{self.telegram_config.get('bot_token')}/sendMessage"
            payload = {
                'chat_id': self.telegram_config.get('chat_id'),
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram notification: {e}")
            
    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        if not self.webhook_config:
            return
            
        try:
            payload = {
                'alert_id': alert.alert_id,
                'type': alert.alert_type.value,
                'severity': alert.severity.value,
                'symbol': alert.symbol,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'data': alert.data
            }
            
            response = requests.post(
                self.webhook_config.get('url'),
                json=payload,
                headers=self.webhook_config.get('headers', {})
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"❌ Failed to send webhook notification: {e}")
            
    def acknowledge_alert(self, alert_id: str, user: str):
        """Acknowledge an alert"""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.now()
                self.active_alerts.remove(alert)
                logger.info(f"✅ Alert {alert_id} acknowledged by {user}")
                break
                
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return self.active_alerts.copy()
        
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        return self.alerts[-limit:] if len(self.alerts) > limit else self.alerts.copy()

class Monitor:
    """Professional monitoring system"""
    
    def __init__(self, portfolio, risk_manager, strategy_manager):
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.strategy_manager = strategy_manager
        self.alert_manager = AlertManager()
        self.monitoring_tasks = {}
        self.is_monitoring = False
        
    def configure_notifications(self, email_config: Dict = None, telegram_config: Dict = None, 
                              webhook_config: Dict = None):
        """Configure notification channels"""
        if email_config:
            self.alert_manager.email_config = email_config
        if telegram_config:
            self.alert_manager.telegram_config = telegram_config
        if webhook_config:
            self.alert_manager.webhook_config = webhook_config
            
    async def start_monitoring(self):
        """Start all monitoring tasks"""
        self.is_monitoring = True
        logger.info("📊 Starting professional monitoring system")
        
        # Start monitoring tasks
        self.monitoring_tasks['risk_monitor'] = asyncio.create_task(self._risk_monitor_loop())
        self.monitoring_tasks['performance_monitor'] = asyncio.create_task(self._performance_monitor_loop())
        self.monitoring_tasks['health_monitor'] = asyncio.create_task(self._health_monitor_loop())
        
    async def stop_monitoring(self):
        """Stop all monitoring tasks"""
        self.is_monitoring = False
        for task in self.monitoring_tasks.values():
            task.cancel()
        logger.info("📊 Stopping monitoring system")
        
    async def _risk_monitor_loop(self):
        """Monitor risk metrics continuously"""
        while self.is_monitoring:
            try:
                # Check portfolio risk
                risk_check, reason = self.risk_manager.update_portfolio_risk()
                
                if not risk_check:
                    self.alert_manager.create_alert(
                        alert_type=AlertType.RISK_THRESHOLD,
                        severity=AlertSeverity.CRITICAL,
                        symbol=None,
                        message=f"Portfolio risk check failed: {reason}",
                        data=self.risk_manager.get_risk_report()
                    )
                    
                # Check individual position risks
                positions = self.portfolio.get_all_positions()
                for symbol, position in positions.items():
                    # Check for high volatility
                    if hasattr(position, 'pnl_percent') and abs(position.pnl_percent) > 10:
                        self.alert_manager.create_alert(
                            alert_type=AlertType.VOLATILITY_SPIKE,
                            severity=AlertSeverity.HIGH,
                            symbol=symbol,
                            message=f"High P&L percentage detected: {position.pnl_percent:.2f}%",
                            data={
                                'pnl_percent': position.pnl_percent,
                                'current_price': position.current_price,
                                'entry_price': position.entry_price
                            }
                        )
                        
                    # Check for position concentration
                    exposure = position.exposure
                    total_value = self.portfolio.calculate_total_value()
                    if total_value > 0:
                        concentration = exposure / total_value
                        if concentration > 0.2:  # 20% concentration threshold
                            self.alert_manager.create_alert(
                                alert_type=AlertType.POSITION_CONCENTRATION,
                                severity=AlertSeverity.MEDIUM,
                                symbol=symbol,
                                message=f"Position concentration too high: {concentration:.2%}",
                                data={
                                    'concentration_percent': concentration,
                                    'position_value': exposure,
                                    'total_portfolio_value': total_value
                                }
                            )
                            
            except Exception as e:
                logger.error(f"❌ Error in risk monitoring: {e}")
                
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def _performance_monitor_loop(self):
        """Monitor performance metrics"""
        previous_metrics = {}
        
        while self.is_monitoring:
            try:
                # Get current metrics
                current_metrics = self.portfolio.get_performance_metrics()
                
                # Compare with previous metrics
                if previous_metrics:
                    # Check for significant drawdown
                    current_dd = current_metrics['current_drawdown_percent']
                    prev_dd = previous_metrics.get('current_drawdown_percent', 0)
                    
                    if current_dd > prev_dd and current_dd > 10:  # 10% drawdown threshold
                        self.alert_manager.create_alert(
                            alert_type=AlertType.DRAWDOWN,
                            severity=AlertSeverity.HIGH,
                            symbol=None,
                            message=f"Significant drawdown detected: {current_dd:.2f}%",
                            data={
                                'current_drawdown': current_dd,
                                'previous_drawdown': prev_dd,
                                'total_pnl': current_metrics['total_pnl']
                            }
                        )
                        
                    # Check for unusual volume changes
                    # This would require additional market data
                    # For now, just log the metrics
                    logger.debug(f"📈 Portfolio metrics: {current_metrics}")
                    
                previous_metrics = current_metrics
                
            except Exception as e:
                logger.error(f"❌ Error in performance monitoring: {e}")
                
            await asyncio.sleep(60)  # Check every minute
            
    async def _health_monitor_loop(self):
        """Monitor system health"""
        while self.is_monitoring:
            try:
                # Check system resources (simplified)
                # In a real system, this would check CPU, memory, disk usage, etc.
                
                # Check if risk manager is functioning
                if not self.risk_manager:
                    self.alert_manager.create_alert(
                        alert_type=AlertType.SYSTEM_HEALTH,
                        severity=AlertSeverity.CRITICAL,
                        symbol=None,
                        message="Risk manager not initialized",
                        data={'component': 'risk_manager', 'status': 'uninitialized'}
                    )
                    
                # Check if portfolio is functioning
                if not self.portfolio:
                    self.alert_manager.create_alert(
                        alert_type=AlertType.SYSTEM_HEALTH,
                        severity=AlertType.CRITICAL,
                        symbol=None,
                        message="Portfolio not initialized",
                        data={'component': 'portfolio', 'status': 'uninitialized'}
                    )
                    
                # Check strategy manager
                if not self.strategy_manager:
                    self.alert_manager.create_alert(
                        alert_type=AlertType.SYSTEM_HEALTH,
                        severity=AlertType.HIGH,
                        symbol=None,
                        message="Strategy manager not initialized",
                        data={'component': 'strategy_manager', 'status': 'uninitialized'}
                    )
                    
            except Exception as e:
                logger.error(f"❌ Error in health monitoring: {e}")
                
            await asyncio.sleep(120)  # Check every 2 minutes
            
    def check_technical_breakouts(self, symbol: str, data: pd.DataFrame):
        """Check for technical breakouts in market data"""
        try:
            if len(data) < 50:
                return
                
            # Identify columns
            col_map = {col.lower(): col for col in data.columns}
            close_col = col_map.get('close', col_map.get('Close', 'close'))
            high_col = col_map.get('high', col_map.get('High', 'high'))
            low_col = col_map.get('low', col_map.get('Low', 'low'))
            
            if close_col not in data.columns:
                return
                
            # Check for breakout above resistance (20-day high)
            recent_high = data[high_col].rolling(20).max().iloc[-2]  # Previous 20-day high
            current_close = data[close_col].iloc[-1]
            
            if current_close > recent_high * 1.01:  # 1% above resistance
                self.alert_manager.create_alert(
                    alert_type=AlertType.TECHNICAL_BREAKOUT,
                    severity=AlertSeverity.MEDIUM,
                    symbol=symbol,
                    message=f"Upward breakout detected: {symbol} above 20-day resistance",
                    data={
                        'current_price': current_close,
                        'resistance_level': recent_high,
                        'breakout_percentage': (current_close - recent_high) / recent_high * 100
                    }
                )
                
            # Check for breakdown below support (20-day low)
            recent_low = data[low_col].rolling(20).min().iloc[-2]  # Previous 20-day low
            if current_close < recent_low * 0.99:  # 1% below support
                self.alert_manager.create_alert(
                    alert_type=AlertType.TECHNICAL_BREAKOUT,
                    severity=AlertSeverity.MEDIUM,
                    symbol=symbol,
                    message=f"Downward breakdown detected: {symbol} below 20-day support",
                    data={
                        'current_price': current_close,
                        'support_level': recent_low,
                        'breakdown_percentage': (current_close - recent_low) / recent_low * 100
                    }
                )
                
        except Exception as e:
            logger.error(f"❌ Error in technical breakout monitoring: {e}")

class DashboardMetrics:
    """Dashboard metrics for monitoring system"""
    
    def __init__(self, portfolio, risk_manager, strategy_manager):
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.strategy_manager = strategy_manager
        
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time dashboard metrics"""
        portfolio_metrics = self.portfolio.get_performance_metrics()
        risk_report = self.risk_manager.get_risk_report()
        strategy_performance = self.strategy_manager.get_strategy_performance() if self.strategy_manager else {}
        
        # Get active alerts
        active_alerts = len(self.risk_manager.risk_events) if hasattr(self.risk_manager, 'risk_events') else 0
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'portfolio': {
                'total_value': portfolio_metrics.get('total_value', 0),
                'total_pnl': portfolio_metrics.get('total_pnl', 0),
                'total_return_percent': portfolio_metrics.get('total_return_percent', 0),
                'current_drawdown_percent': portfolio_metrics.get('current_drawdown_percent', 0),
                'portfolio_exposure_percent': portfolio_metrics.get('portfolio_exposure_percent', 0),
                'number_of_positions': portfolio_metrics.get('number_of_positions', 0),
                'sharpe_ratio': portfolio_metrics.get('sharpe_ratio', 0)
            },
            'risk': {
                'circuit_breaker_active': risk_report['current_risk_status'].get('circuit_breaker_active', False),
                'max_drawdown_limit': risk_report['risk_limits'].get('max_drawdown', 0),
                'current_drawdown': risk_report['current_risk_status'].get('current_drawdown', 0),
                'portfolio_exposure_limit': risk_report['risk_limits'].get('max_portfolio_risk', 0),
                'current_exposure': risk_report['current_risk_status'].get('portfolio_exposure', 0),
                'active_alerts': active_alerts
            },
            'strategies': {
                'active_strategies': len(strategy_performance),
                'combined_performance': strategy_performance
            },
            'system': {
                'uptime': self._calculate_uptime(),
                'last_heartbeat': datetime.now().isoformat()
            }
        }
        
        return metrics
        
    def _calculate_uptime(self) -> str:
        """Calculate system uptime"""
        # In a real system, this would track actual uptime
        # For now, return a placeholder
        return "System Online"
        
    def get_historical_performance(self, days: int = 30) -> pd.DataFrame:
        """Get historical performance data for charts"""
        # This would typically query a database for historical data
        # For now, return a mock dataframe
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        returns = [0.0] * days  # Mock returns
        
        df = pd.DataFrame({
            'date': dates,
            'portfolio_value': [self.portfolio.initial_capital] * days,  # Mock values
            'daily_return': returns,
            'drawdown': [0.0] * days
        })
        
        return df

# Global monitoring instance
monitor = None
dashboard_metrics = None

def initialize_monitoring_system(portfolio, risk_manager, strategy_manager) -> tuple:
    """Initialize global monitoring system"""
    global monitor, dashboard_metrics
    
    monitor = Monitor(portfolio, risk_manager, strategy_manager)
    dashboard_metrics = DashboardMetrics(portfolio, risk_manager, strategy_manager)
    
    logger.info("📊 Monitoring system initialized with real-time alerts")
    return monitor, dashboard_metrics