#!/usr/bin/env python3
"""
Execution Engine Demo for Chloe AI 0.4
Demonstrates professional order execution with smart routing and impact modeling
"""

import asyncio
import logging
from datetime import datetime
from execution_engine import get_execution_engine, Order

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_execution_engine():
    """Demonstrate professional execution engine capabilities"""
    logger.info("⚡ EXECUTION ENGINE DEMO")
    logger.info("=" * 50)
    
    try:
        # Initialize execution engine
        logger.info("🔧 Initializing Execution Engine...")
        engine = get_execution_engine(initial_capital=100000.0)
        logger.info(f"✅ Engine initialized with ${engine.current_capital:,.2f}")
        
        # Create sample orders
        sample_orders = [
            Order(
                order_id="BTC_BUY_001",
                symbol="BTC/USDT",
                side="BUY",
                quantity=0.5,  # 0.5 BTC
                price=48500,
                order_type="LIMIT"
            ),
            Order(
                order_id="ETH_SELL_001",
                symbol="ETH/USDT",
                side="SELL",
                quantity=5.0,  # 5 ETH
                price=3650,
                order_type="MARKET"
            ),
            Order(
                order_id="SOL_BUY_001",
                symbol="SOL/USDT",
                side="BUY",
                quantity=100.0,  # 100 SOL (larger order)
                price=47.5,
                order_type="LIMIT"
            )
        ]
        
        # Execute orders with different strategies
        logger.info(f"\n🚀 Executing {len(sample_orders)} sample orders...")
        
        execution_reports = []
        
        for i, order in enumerate(sample_orders):
            strategy = ['SMART', 'AGGRESSIVE', 'SMART'][i]  # Different strategies
            logger.info(f"\n📝 Order {i+1}: {order.symbol} {order.side} {order.quantity}")
            logger.info(f"   Strategy: {strategy}")
            
            # Execute order
            report = await engine.execute_order(order, execution_strategy=strategy)
            execution_reports.append(report)
            
            # Show execution details
            logger.info(f"   Status: {report.status}")
            logger.info(f"   Executed Quantity: {report.executed_quantity:.4f}")
            logger.info(f"   Average Price: ${report.average_price:.2f}")
            logger.info(f"   Slippage: {report.slippage:.4f}")
            logger.info(f"   Fees: ${report.fees:.2f}")
            logger.info(f"   Execution Time: {report.execution_time:.2f}s")
        
        # Show performance summary
        logger.info(f"\n📊 EXECUTION PERFORMANCE SUMMARY:")
        performance = engine.get_performance_summary()
        
        logger.info(f"   Total Orders: {performance['total_orders']}")
        logger.info(f"   Filled Orders: {performance['filled_orders']}")
        logger.info(f"   Fill Rate: {performance['fill_rate']:.2%}")
        logger.info(f"   Average Slippage: {performance['average_slippage']:.4f}")
        logger.info(f"   Average Market Impact: {performance['average_market_impact']:.4f}")
        logger.info(f"   Total Fees Paid: ${performance['total_fees_paid']:.2f}")
        logger.info(f"   Current Capital: ${performance['current_capital']:,.2f}")
        logger.info(f"   Capital Utilization: {performance['capital_utilization']:.2%}")
        
        # Show detailed execution reports
        logger.info(f"\n📋 DETAILED EXECUTION REPORTS:")
        for i, report in enumerate(execution_reports):
            logger.info(f"   {i+1}. {report.symbol} ({report.order_id}):")
            logger.info(f"      Strategy: {['SMART', 'AGGRESSIVE', 'SMART'][i]}")
            logger.info(f"      Status: {report.status}")
            logger.info(f"      Quantity: {report.planned_quantity:.4f} → {report.executed_quantity:.4f}")
            logger.info(f"      Price: ${report.average_price:.2f}")
            logger.info(f"      Slippage: {report.slippage:.2%}")
            logger.info(f"      Impact: {report.market_impact:.2%}")
            logger.info(f"      Fees: ${report.fees:.2f}")
        
        logger.info(f"\n{'='*50}")
        logger.info("✅ EXECUTION ENGINE DEMO COMPLETED")
        logger.info("🚀 Key achievements:")
        logger.info("   • Implemented professional order execution engine")
        logger.info("   • Smart order routing with venue selection")
        logger.info("   • Market impact and slippage modeling")
        logger.info("   • Multiple execution strategies (SMART/AGGRESSIVE/PASSIVE)")
        logger.info("   • Real-time performance monitoring")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_order_slicing():
    """Demonstrate large order slicing capabilities"""
    logger.info(f"\n🔪 LARGE ORDER SLICING DEMO")
    logger.info("=" * 40)
    
    try:
        from execution_engine import OrderRouter
        
        router = OrderRouter()
        
        # Create large order
        large_order = Order(
            order_id="LARGE_BTC_ORDER",
            symbol="BTC/USDT",
            side="BUY",
            quantity=10.0,  # 10 BTC (large order)
            price=48500
        )
        
        logger.info(f"Original order: {large_order.quantity} {large_order.symbol}")
        
        # Slice the order
        chunks = router.slice_large_order(large_order, max_participation_rate=0.02)
        
        logger.info(f"Sliced into {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            logger.info(f"   Chunk {i+1}: {chunk.quantity:.4f} {chunk.symbol}")
        
        total_quantity = sum(chunk.quantity for chunk in chunks)
        logger.info(f"Total quantity preserved: {total_quantity:.4f} (original: {large_order.quantity})")
        
        logger.info("✅ Order slicing demonstrated successfully")
        
    except Exception as e:
        logger.error(f"❌ Slicing demo failed: {e}")

def demonstrate_market_impact():
    """Demonstrate market impact modeling"""
    logger.info(f"\n📈 MARKET IMPACT MODELING DEMO")
    logger.info("=" * 40)
    
    try:
        from execution_engine import MarketImpactModel, SlippageEstimator
        
        impact_model = MarketImpactModel()
        slippage_model = SlippageEstimator()
        
        # Test different order sizes
        test_cases = [
            ("BTC/USDT", 0.1, 20000000000, 48500),    # Small order
            ("BTC/USDT", 1.0, 20000000000, 48500),    # Medium order  
            ("BTC/USDT", 10.0, 20000000000, 48500),   # Large order
            ("SOL/USDT", 1000.0, 2000000000, 47.5),   # Large altcoin order
        ]
        
        logger.info("Market Impact Analysis:")
        for symbol, qty, volume, price in test_cases:
            impact_pct = impact_model.estimate_impact(symbol, qty, volume, price)
            slippage_pct = slippage_model.estimate_slippage(symbol, qty, 0.03)  # 3% volatility
            
            order_value = qty * price
            participation_rate = (order_value / volume) * 100
            
            logger.info(f"   {symbol}: {qty} units (${order_value:,.0f})")
            logger.info(f"      Participation Rate: {participation_rate:.4f}%")
            logger.info(f"      Estimated Impact: {impact_pct:.4f} ({impact_pct*100:.2f}%)")
            logger.info(f"      Estimated Slippage: {slippage_pct:.4f} ({slippage_pct*100:.2f}%)")
            logger.info()
        
        logger.info("✅ Market impact modeling demonstrated successfully")
        
    except Exception as e:
        logger.error(f"❌ Impact modeling demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe AI 0.4 - Professional Execution Engine Demo")
    print("Institutional-grade order execution and routing")
    print()
    
    # Run main execution demo
    await demonstrate_execution_engine()
    
    # Run additional demonstrations
    demonstrate_order_slicing()
    demonstrate_market_impact()
    
    print(f"\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY")
    print("Chloe AI now has professional execution capabilities!")

if __name__ == "__main__":
    asyncio.run(main())