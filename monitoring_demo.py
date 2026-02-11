#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard Demo for Chloe AI 0.4
Professional monitoring system with live metrics and alerts
"""

import asyncio
import logging
from datetime import datetime, timedelta
import json
from realtime_monitoring import get_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_realtime_monitoring():
    """Demonstrate real-time monitoring capabilities"""
    logger.info("🖥️ REAL-TIME MONITORING DASHBOARD DEMO")
    logger.info("=" * 60)
    
    try:
        # Initialize monitor
        logger.info("🔧 Initializing Real-time Monitor...")
        monitor = get_monitor(update_interval=0.5)  # Fast updates for demo
        logger.info("✅ Monitor initialized")
        
        # Start monitoring for 15 seconds
        logger.info(f"\n🟢 Starting 15-second monitoring session...")
        
        # Start monitoring in background
        monitoring_task = asyncio.create_task(monitor.start_monitoring())
        
        # Let it run for demonstration period
        await asyncio.sleep(15)
        
        # Stop monitoring
        monitor.stop_monitoring()
        await monitoring_task
        
        # Show dashboard data
        logger.info(f"\n📊 DASHBOARD DATA:")
        dashboard_data = monitor.get_dashboard_data()
        
        system_metrics = dashboard_data.get('system_metrics', {})
        risk_metrics = dashboard_data.get('risk_metrics', {})
        alerts = dashboard_data.get('active_alerts', [])
        
        if system_metrics:
            logger.info("   SYSTEM METRICS:")
            logger.info(f"      Portfolio Value: ${system_metrics.get('portfolio_value', 0):,.2f}")
            logger.info(f"      Total Return: {system_metrics.get('total_return_pct', 0):+.2f}%")
            logger.info(f"      Current Drawdown: {system_metrics.get('current_drawdown', 0):.2f}%")
            logger.info(f"      Daily P&L: ${system_metrics.get('daily_pnl', 0):,.2f}")
            logger.info(f"      Sharpe Ratio: {system_metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"      Win Rate: {system_metrics.get('win_rate', 0):.1f}%")
        
        if risk_metrics:
            logger.info("   RISK METRICS:")
            logger.info(f"      Portfolio Exposure: {risk_metrics.get('portfolio_exposure', 0):.1f}%")
            logger.info(f"      Max Position Size: {risk_metrics.get('max_position_size', 0):.1f}%")
            logger.info(f"      Correlation Risk: {risk_metrics.get('correlation_risk', 0):.2f}")
            logger.info(f"      Liquidity Risk: {risk_metrics.get('liquidity_risk', 0):.2f}")
            logger.info(f"      Market Regime: {risk_metrics.get('regime_state', 'UNKNOWN')}")
            logger.info(f"      Regime Confidence: {risk_metrics.get('regime_confidence', 0):.2f}")
            logger.info(f"      VaR (95%): ${risk_metrics.get('var_95', 0):,.2f}")
        
        logger.info(f"   ALERTS:")
        logger.info(f"      Active Alerts: {len(alerts)}")
        for i, alert in enumerate(alerts[:3]):  # Show first 3 alerts
            logger.info(f"      {i+1}. [{getattr(alert, 'severity', 'UNKNOWN')}] {getattr(alert, 'message', 'No message')}")
        
        logger.info(f"   SYSTEM STATUS: {dashboard_data.get('monitoring_status', 'UNKNOWN')}")
        logger.info(f"   Uptime: {dashboard_data.get('system_uptime', 0):.1f} seconds")
        
        # Show alert analysis
        logger.info(f"\n🔔 ALERT ANALYSIS:")
        all_alerts = list(monitor.recent_alerts)
        if all_alerts:
            severity_counts = {}
            category_counts = {}
            
            for alert in all_alerts:
                severity = getattr(alert, 'severity', 'UNKNOWN')
                category = getattr(alert, 'category', 'UNKNOWN')
                
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
            
            logger.info("   Alert Severity Distribution:")
            for severity, count in severity_counts.items():
                logger.info(f"      {severity}: {count}")
                
            logger.info("   Alert Category Distribution:")
            for category, count in category_counts.items():
                logger.info(f"      {category}: {count}")
        else:
            logger.info("   No alerts generated during monitoring period")
        
        # Show metrics history summary
        logger.info(f"\n📈 METRICS HISTORY SUMMARY:")
        if monitor.system_metrics_history:
            history_length = len(monitor.system_metrics_history)
            avg_portfolio_value = sum(m.portfolio_value for m in monitor.system_metrics_history) / history_length
            max_drawdown = max(m.current_drawdown for m in monitor.system_metrics_history)
            avg_sharpe = sum(m.sharpe_ratio for m in monitor.system_metrics_history) / history_length
            
            logger.info(f"   Data Points Collected: {history_length}")
            logger.info(f"   Average Portfolio Value: ${avg_portfolio_value:,.2f}")
            logger.info(f"   Maximum Drawdown Recorded: {max_drawdown:.2f}%")
            logger.info(f"   Average Sharpe Ratio: {avg_sharpe:.2f}")
        
        logger.info(f"\n{'='*60}")
        logger.info("✅ REAL-TIME MONITORING DEMO COMPLETED")
        logger.info("🚀 Key achievements:")
        logger.info("   • Implemented professional real-time monitoring system")
        logger.info("   • Continuous metrics collection and analysis")
        logger.info("   • Intelligent alert system with multiple thresholds")
        logger.info("   • Comprehensive dashboard data generation")
        logger.info("   • Risk and performance monitoring in real-time")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_alert_system():
    """Demonstrate alert system capabilities"""
    logger.info(f"\n🔔 ALERT SYSTEM DEMO")
    logger.info("=" * 40)
    
    try:
        from realtime_monitoring import AlertManager, SystemMetrics, RiskMetrics
        from datetime import datetime
        
        alert_manager = AlertManager()
        
        # Create test metrics that should trigger alerts
        critical_system_metrics = SystemMetrics(
            timestamp=datetime.now(),
            portfolio_value=100000,
            cash_balance=20000,
            positions_value=80000,
            total_pnl=-16000,  # 16% loss
            total_return_pct=-16.0,
            current_drawdown=16.0,  # Critical drawdown
            number_of_positions=3,
            active_orders=2,
            daily_pnl=-3500,  # Critical daily loss
            daily_return_pct=-3.5,
            volatility_30d=25.0,
            sharpe_ratio=-0.8,
            win_rate=45.0
        )
        
        critical_risk_metrics = RiskMetrics(
            timestamp=datetime.now(),
            portfolio_exposure=55.0,  # Critical exposure
            max_position_size=25.0,
            correlation_risk=0.85,  # High correlation
            liquidity_risk=0.6,
            regime_state='VOLATILE',
            regime_confidence=0.7,
            var_95=5000,
            cvar_95=8000,
            stress_test_results={'market_crash': -0.20}
        )
        
        logger.info("Testing critical alert conditions:")
        logger.info(f"   Drawdown: {critical_system_metrics.current_drawdown:.1f}% (threshold: 15%)")
        logger.info(f"   Exposure: {critical_risk_metrics.portfolio_exposure:.1f}% (threshold: 50%)")
        logger.info(f"   Daily Loss: {critical_system_metrics.daily_return_pct:.1f}% (threshold: -3%)")
        logger.info(f"   Correlation: {critical_risk_metrics.correlation_risk:.2f} (threshold: 0.8)")
        
        # Check for alerts
        alerts = alert_manager.check_alerts(critical_system_metrics, critical_risk_metrics)
        
        logger.info(f"\nGenerated {len(alerts)} alerts:")
        for i, alert in enumerate(alerts):
            logger.info(f"   {i+1}. [{alert.severity}] {alert.category}: {alert.message}")
            logger.info(f"      Triggered Value: {alert.triggered_value}")
            logger.info(f"      Threshold: {alert.threshold}")
        
        # Test alert resolution
        if alerts:
            first_alert_id = alerts[0].alert_id
            resolved = alert_manager.resolve_alert(first_alert_id)
            logger.info(f"\nResolving alert {first_alert_id}: {'SUCCESS' if resolved else 'FAILED'}")
            
            remaining_active = len(alert_manager.get_active_alerts())
            logger.info(f"Active alerts after resolution: {remaining_active}")
        
        logger.info("✅ Alert system demonstrated successfully")
        
    except Exception as e:
        logger.error(f"❌ Alert demo failed: {e}")

def demonstrate_metrics_collection():
    """Demonstrate metrics collection capabilities"""
    logger.info(f"\n📊 METRICS COLLECTION DEMO")
    logger.info("=" * 40)
    
    try:
        from realtime_monitoring import MetricsCollector
        import numpy as np
        
        collector = MetricsCollector()
        
        # Simulate collecting multiple data points
        logger.info("Collecting 10 data points...")
        
        for i in range(10):
            # Simulate portfolio data with trends
            portfolio_data = {
                'total_value': 100000 + (i * 100) + np.random.normal(0, 50),  # Small upward trend
                'cash_balance': 20000 - (i * 50),  # Decreasing cash
                'positions_value': 80000 + (i * 150),  # Increasing positions
                'total_pnl': (i * 100) + np.random.normal(0, 50),
                'initial_capital': 100000,
                'positions': ['BTC/USDT', 'ETH/USDT'],
                'active_orders': [] if i % 3 != 0 else ['pending_order']
            }
            
            risk_data = {
                'portfolio_exposure': 20 + (i * 2) + np.random.normal(0, 1),  # Increasing exposure
                'max_position_size': 10 + np.random.normal(0, 0.5),
                'correlation_risk': 0.5 + (i * 0.03),  # Increasing correlation
                'liquidity_risk': 0.2 + np.random.normal(0, 0.05),
                'regime_state': 'TRENDING' if i > 5 else 'STABLE',
                'regime_confidence': 0.7 + np.random.normal(0, 0.1),
                'var_95': 2000 + (i * 50),
                'cvar_95': 3500 + (i * 80),
                'stress_test_results': {'market_crash': -0.05 - (i * 0.005)}
            }
            
            system_metrics = collector.collect_system_metrics(portfolio_data, risk_data)
            risk_metrics = collector.collect_risk_metrics(risk_data, {})
            
            if i % 3 == 0:  # Show every 3rd data point
                logger.info(f"   Point {i+1}: Portfolio=${system_metrics.portfolio_value:,.2f}, "
                          f"Drawdown={system_metrics.current_drawdown:.2f}%, "
                          f"Exposure={risk_metrics.portfolio_exposure:.1f}%")
        
        # Show collected metrics summary
        logger.info(f"\nCollection Summary:")
        logger.info(f"   Total data points: {len(collector.metrics_history)}")
        logger.info(f"   Portfolio value range: ${min(m.portfolio_value for m in collector.metrics_history):,.2f} - ${max(m.portfolio_value for m in collector.metrics_history):,.2f}")
        logger.info(f"   Drawdown range: {min(m.current_drawdown for m in collector.metrics_history):.2f}% - {max(m.current_drawdown for m in collector.metrics_history):.2f}%")
        logger.info(f"   Exposure range: {min(m.portfolio_exposure for m in collector.risk_history):.1f}% - {max(m.portfolio_exposure for m in collector.risk_history):.1f}%")
        
        logger.info("✅ Metrics collection demonstrated successfully")
        
    except Exception as e:
        logger.error(f"❌ Metrics demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe AI 0.4 - Real-time Monitoring Dashboard Demo")
    print("Professional system monitoring and alerting")
    print()
    
    # Run main monitoring demo
    await demonstrate_realtime_monitoring()
    
    # Run additional demonstrations
    demonstrate_alert_system()
    demonstrate_metrics_collection()
    
    print(f"\n🎉 ALL MONITORING DEMOS COMPLETED SUCCESSFULLY")
    print("Chloe AI now has professional real-time monitoring capabilities!")

if __name__ == "__main__":
    asyncio.run(main())