"""
Institutional Trading Platform - Main Orchestrator
Portfolio-first architecture with event-driven execution
"""

import asyncio
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Import core components
from core.event_bus import EventBus, event_bus, MarketEvent, SignalEvent, OrderEvent, FillEvent, ForecastEvent, Event, EventType
from portfolio.portfolio import Portfolio, initialize_portfolio, portfolio
from risk.risk_engine import RiskManager, RiskLimits, initialize_risk_manager, risk_manager
from execution.order_manager import OrderManager, initialize_order_manager, order_manager
from backtest.engine import BacktestEngine
from data.pipeline import DataPipeline, initialize_data_pipeline, data_pipeline
from strategies.advanced_strategies import StrategyManager, EnhancedStrategyManager, initialize_strategy_manager, strategy_manager
from monitoring.alerts import initialize_monitoring_system, monitor, dashboard_metrics

# Import new market data gateway
from market_data.gateway import MarketDataGatewayService, initialize_market_data_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class TradingOrchestrator:
    """Main institutional trading platform orchestrator"""
    
    def __init__(self, initial_capital: float = 100000.0, mode: str = 'paper'):
        self.mode = mode
        self.initial_capital = initial_capital
        self.is_running = False
        
        # Initialize core components
        self._initialize_components()
        
        # Subscribe to events
        self._setup_event_handlers()
        
    def _initialize_components(self):
        """Initialize all system components"""
        logger.info("🚀 Initializing Institutional Trading Platform...")
        
        # Initialize portfolio
        initialize_portfolio(self.initial_capital)
        
        # Initialize risk manager
        risk_limits = RiskLimits(
            max_risk_per_trade=0.01,      # 1% per trade
            max_portfolio_risk=0.05,      # 5% total exposure
            max_drawdown=0.15,            # 15% max drawdown
            max_daily_loss=0.03,          # 3% daily loss limit
            max_position_concentration=0.25  # 25% per position
        )
        initialize_risk_manager(portfolio, risk_limits)
        
        # Initialize order manager
        initialize_order_manager()
        
        # Initialize data pipeline
        initialize_data_pipeline()
        
        # Initialize strategy manager
        initialize_strategy_manager()
        
        # Initialize forecast service
        from services.forecast.forecast_service import ForecastService
        self.forecast_service = ForecastService()
        
        # Initialize market data gateway service
        if self.mode != 'backtest':
            self.market_data_service = initialize_market_data_service(
                symbols=['BTCUSDT', 'ETHUSDT']  # Default symbols
            )
        
        # Initialize monitoring system
        initialize_monitoring_system(portfolio, risk_manager, strategy_manager)
        
        logger.info("✅ All components initialized successfully")
        
    def _setup_event_handlers(self):
        """Setup event-driven architecture"""
        # Market data handler
        event_bus.subscribe(EventType.MARKET, self._handle_market_event)
        
        # Signal handler
        event_bus.subscribe(EventType.SIGNAL, self._handle_signal_event)
        
        # Order handler
        event_bus.subscribe(EventType.ORDER, self._handle_order_event)
        
        # Fill handler
        event_bus.subscribe(EventType.FILL, self._handle_fill_event)
        
        # Risk handler
        event_bus.subscribe(EventType.RISK, self._handle_risk_event)
        
        # Forecast handler
        event_bus.subscribe(EventType.FORECAST, self._handle_forecast_event)
        
        logger.info("🔌 Event handlers connected")
        
    async def _handle_market_event(self, event: MarketEvent):
        """Handle market data updates"""
        logger.debug(f"📊 Market update: {event.symbol} @ ${event.price}")
        
        # Update portfolio with new prices
        portfolio.update_market_prices({event.symbol: event.price})
        
        # Update risk manager
        risk_check_passed, reason = risk_manager.update_portfolio_risk()
        if not risk_check_passed:
            # Publish risk event
            risk_event = Event(
                event_type=EventType.RISK,
                timestamp=datetime.now(),
                data={'reason': reason, 'action': 'PORTFOLIO_RISK_CHECK'}
            )
            await event_bus.publish(risk_event)
            
        # Check for technical breakouts
        # In a real system, this would use actual market data
        # monitor.check_technical_breakouts(event.symbol, None)
            
    async def _handle_signal_event(self, event: SignalEvent):
        """Handle trading signals - Portfolio-first approach"""
        logger.info(f"🎯 Signal received: {event.symbol} {event.signal} ({event.confidence:.2f})")
        
        # Step 1: Portfolio analysis (Portfolio-first!)
        current_positions = portfolio.get_all_positions()
        position_exists = event.symbol in current_positions
        
        # Step 2: Risk assessment
        market_price = await order_manager.get_market_price(event.symbol)
        if market_price <= 0:
            logger.warning(f"⚠️ Invalid price for {event.symbol}")
            return
            
        # Calculate ATR or use volatility for risk sizing
        atr = market_price * 0.02  # Simplified ATR calculation
        
        # Check if trade complies with risk limits
        risk_check_passed, reason = risk_manager.check_trade_risk(
            symbol=event.symbol,
            signal=event.signal,
            quantity=1.0,  # Placeholder, will be calculated
            entry_price=market_price,
            atr=atr
        )
        
        if not risk_check_passed:
            logger.warning(f"❌ Risk check failed: {reason}")
            return
            
        # Step 3: Calculate position size
        position_size = risk_manager.calculate_position_size(
            symbol=event.symbol,
            signal=event.signal,
            entry_price=market_price,
            atr=atr
        )
        
        if abs(position_size) < 0.001:  # Minimum position size
            logger.warning(f"⚠️ Position size too small: {position_size}")
            return
            
        # Step 4: Generate order event
        order_event = OrderEvent(
            event_type=EventType.ORDER,
            timestamp=datetime.now(),
            data={
                'strategy_id': event.strategy_id,
                'confidence': event.confidence
            },
            symbol=event.symbol,
            order_type='MARKET',
            side=event.signal,
            quantity=position_size,
            price=market_price
        )
        
        await event_bus.publish(order_event)
        
    async def _handle_order_event(self, event: OrderEvent):
        """Handle order execution requests"""
        logger.info(f"📋 Order request: {event.symbol} {event.side} {event.quantity}")
        
        # Place order through order manager
        broker_name = 'paper' if self.mode == 'paper' else 'default'
        order = await order_manager.place_order(
            symbol=event.symbol,
            side=event.side,
            quantity=event.quantity,
            order_type=event.order_type,
            price=event.price,
            broker=broker_name
        )
        
        if order:
            logger.info(f"✅ Order submitted: {order.order_id}")
        else:
            logger.error(f"❌ Order placement failed for {event.symbol}")
            
    async def _handle_fill_event(self, event: FillEvent):
        """Handle order fill confirmations"""
        logger.info(f"💰 Fill confirmed: {event.symbol} {event.side} {event.quantity}@${event.price}")
        
        # Update portfolio based on fill
        if event.side == 'BUY':
            portfolio.enter_position(
                symbol=event.symbol,
                quantity=event.quantity,
                price=event.price,
                commission=event.commission
            )
        else:  # SELL
            portfolio.exit_position(
                symbol=event.symbol,
                quantity=event.quantity,
                price=event.price,
                commission=event.commission
            )
            
    async def _handle_forecast_event(self, event: ForecastEvent):
        """Handle forecast events - adjust strategy based on market predictions"""
        logger.info(f"🔮 Forecast received: {event.symbol} E[R]={event.expected_return:.4f} (conf: {event.confidence:.2f})")
        
        # Update strategy based on forecast
        # This would typically trigger position sizing adjustments
        # based on the forecast confidence and expected return
        
        # Example: adjust position size based on forecast
        if event.confidence > 0.7 and abs(event.expected_return) > 0.001:
            # High confidence forecast - adjust positions
            logger.info(f"📈 High confidence forecast - adjusting positions for {event.symbol}")
        elif event.confidence < 0.4:
            # Low confidence forecast - reduce exposure
            logger.info(f"📉 Low confidence forecast - reducing exposure to {event.symbol}")
        
    async def _handle_risk_event(self, event: Event):
        """Handle risk management events"""
        logger.critical(f"🚨 Risk event: {event.data}")
        
        if event.data.get('action') == 'PORTFOLIO_RISK_CHECK':
            reason = event.data.get('reason')
            if 'MAX_DRAWDOWN_EXCEEDED' in reason or 'CONSECUTIVE_LOSSES' in reason:
                logger.critical("🛑 Emergency risk protocols activated")
                # Risk manager will handle circuit breaker automatically
                
    async def start_trading(self, symbols: List[str]):
        """Start live trading system"""
        logger.info(f"🚀 Starting trading for symbols: {symbols}")
        
        # Start monitoring system
        await monitor.start_monitoring()
        
        # Start event bus processing
        event_task = asyncio.create_task(event_bus.process_events())
        
        # Initialize and start market data gateway if not in backtest mode
        market_data_task = None
        if self.mode in ['paper', 'live']:
            try:
                # Initialize market data service with the provided symbols
                self.market_data_service = initialize_market_data_service(
                    symbols=symbols
                )
                
                # Start market data gateway
                market_data_task = asyncio.create_task(
                    self.market_data_service.start(symbols)
                )
                logger.info("📡 Market data gateway started")
            except Exception as e:
                logger.error(f"❌ Failed to start market data gateway: {e}, falling back to simulator")
        
        # Start market data feed simulation if gateway failed or in backtest mode
        market_sim_task = None
        if self.mode == 'backtest' or market_data_task is None:
            market_sim_task = asyncio.create_task(self._market_data_simulator(symbols))
        
        # Start strategy processing
        strategy_task = asyncio.create_task(self._strategy_processor(symbols))
        
        self.is_running = True
        
        try:
            # Keep system running
            while self.is_running:
                await asyncio.sleep(1)
                # Update order statuses
                await order_manager.update_order_status()
                
                # Print dashboard metrics periodically
                if datetime.now().second % 10 == 0:  # Every 10 seconds
                    metrics = dashboard_metrics.get_real_time_metrics()
                    logger.info(f"📈 Portfolio Value: ${metrics['portfolio']['total_value']:,.2f}, "
                               f"P&L: ${metrics['portfolio']['total_pnl']:,.2f}, "
                               f"Drawdown: {metrics['portfolio']['current_drawdown_percent']:.2f}%")
                
        except KeyboardInterrupt:
            logger.info("🛑 Trading system stopped by user")
        finally:
            self.is_running = False
            event_bus.stop()
            await event_task
            if market_data_task:
                market_data_task.cancel()
            if market_sim_task:
                market_sim_task.cancel()
            strategy_task.cancel()
            await monitor.stop_monitoring()
            
    async def _market_data_simulator(self, symbols: List[str]):
        """Simulate market data feed (would connect to real feeds in production)"""
        while self.is_running:
            try:
                for symbol in symbols:
                    # Simulate price updates (random walk)
                    current_price = await order_manager.get_market_price(symbol)
                    if current_price > 0:
                        # Small random price movement
                        price_change = np.random.normal(0, 0.005)  # 0.5% volatility
                        new_price = current_price * (1 + price_change)
                        
                        # Publish market event
                        market_event = MarketEvent(
                            event_type=EventType.MARKET,
                            timestamp=datetime.now(),
                            data={'source': 'simulator'},
                            symbol=symbol,
                            price=new_price,
                            volume=np.random.randint(1000, 10000),
                            exchange='SIMULATED'
                        )
                        await event_bus.publish(market_event)
                        
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Market simulator error: {e}")
                await asyncio.sleep(10)
                
    async def _strategy_processor(self, symbols: List[str]):
        """Process trading strategies and generate signals"""
        while self.is_running:
            try:
                for symbol in symbols:
                    # In a real system, this would fetch real market data
                    # For demo, we'll simulate signals
                    await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
                    
                    # Generate signals from strategies
                    # In a real system, we would fetch real data and process it
                    # For now, simulate signals randomly
                    if np.random.random() > 0.7:  # 30% chance of signal
                        signal_type = 'BUY' if np.random.random() > 0.5 else 'SELL'
                        confidence = np.random.uniform(0.6, 0.9)
                        
                        signal_event = SignalEvent(
                            event_type=EventType.SIGNAL,
                            timestamp=datetime.now(),
                            data={'source': 'demo_strategy'},
                            symbol=symbol,
                            signal=signal_type,
                            confidence=confidence,
                            strategy_id='demo_strategy'
                        )
                        await event_bus.publish(signal_event)
                        
                await asyncio.sleep(10)  # Process strategies every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Strategy processor error: {e}")
                await asyncio.sleep(15)
                
    async def run_backtest(self, symbol: str, days: int = 365) -> Dict:
        """Run professional backtest"""
        logger.info(f"🔄 Running backtest for {symbol}")
        
        # Get historical data
        data = await self._fetch_data_for_backtest(
            symbol=symbol, 
            data_type='crypto_ohlcv',
            start_date=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        )
        
        if data is None or len(data) < 50:
            return {"error": "Insufficient data for backtesting"}
            
        # Initialize backtest engine
        backtest_engine = BacktestEngine(initial_capital=self.initial_capital)
        
        # Simple moving average strategy for demo
        def demo_strategy(symbol, timestamp, row, positions):
            if 'sma_20' not in row or 'sma_50' not in row:
                return 'HOLD', 0.0
                
            if row['sma_20'] > row['sma_50'] and symbol not in positions:
                return 'BUY', 0.8
            elif row['sma_20'] < row['sma_50'] and symbol in positions:
                return 'SELL', 0.8
            return 'HOLD', 0.0
            
        # Run backtest
        results = backtest_engine.run_backtest(data, demo_strategy, [symbol])
        
        logger.info(f"✅ Backtest completed for {symbol}")
        return results
        
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        portfolio_metrics = portfolio.get_performance_metrics() if portfolio else {}
        risk_report = risk_manager.get_risk_report() if risk_manager else {}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'RUNNING' if self.is_running else 'STOPPED',
            'mode': self.mode,
            'portfolio': portfolio_metrics,
            'risk': risk_report,
            'active_orders': len(order_manager.get_active_orders()) if order_manager else 0,
            'total_trades': len(order_manager.get_order_history()) if order_manager else 0
        }
        
    async def _fetch_data_for_backtest(self, symbol: str, data_type: str = 'crypto_ohlcv',
                                start_date: str = None, end_date: str = None,
                                create_features: bool = True) -> pd.DataFrame:
        """Helper method to fetch data for backtesting"""
        # Create synthetic test data
        try:
            from datetime import datetime, timedelta
            dates = pd.date_range(start=start_date or (datetime.now() - timedelta(days=100)), periods=100, freq='D')
            data = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 40000,
                'high': np.random.randn(100).cumsum() + 40000,
                'low': np.random.randn(100).cumsum() + 40000,
                'close': np.random.randn(100).cumsum() + 40000,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
            
            if create_features:
                # Create basic features
                data['returns'] = data['close'].pct_change()
                data['sma_20'] = data['close'].rolling(window=20).mean()
                data['sma_50'] = data['close'].rolling(window=50).mean()
                data['volatility_20'] = data['returns'].rolling(window=20).std()
                
            return data
        except Exception as e:
            logger.error(f"❌ Error creating test data: {e}")
            return None

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Institutional Trading Platform')
    parser.add_argument('--mode', choices=['paper', 'live', 'backtest'], default='paper',
                       help='Trading mode')
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                       help='Trading symbols')
    parser.add_argument('--days', type=int, default=365,
                       help='Days of historical data for backtest')
    
    args = parser.parse_args()
    
    logger.info("🏛️ Institutional Trading Platform Starting")
    logger.info(f"Mode: {args.mode} | Capital: ${args.capital:,.2f}")
    
    # Initialize orchestrator
    orchestrator = TradingOrchestrator(
        initial_capital=args.capital,
        mode=args.mode
    )
    
    try:
        if args.mode == 'backtest':
            # Run backtest for each symbol
            for symbol in args.symbols:
                results = await orchestrator.run_backtest(symbol, args.days)
                print(f"\n📊 Backtest Results for {symbol}:")
                print(f"Total Return: {results['summary']['total_return_percent']:.2f}%")
                print(f"Sharpe Ratio: {results['risk_metrics']['sharpe_ratio']:.2f}")
                print(f"Max Drawdown: {results['risk_metrics']['max_drawdown_percent']:.2f}%")
                print(f"Win Rate: {results['summary']['win_rate_percent']:.2f}%")
                
        else:
            # Start live trading
            await orchestrator.start_trading(args.symbols)
            
    except KeyboardInterrupt:
        logger.info("🛑 System shutdown requested")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        raise
    finally:
        logger.info("👋 Trading platform shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())