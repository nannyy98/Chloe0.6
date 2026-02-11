#!/usr/bin/env python3
"""
Risk-First Architecture Demo for Chloe AI 0.4
Demonstrates the professional trading AI architecture where Risk Engine orchestrates decisions
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_market_data():
    """Create sample market data for demonstration"""
    logger.info("📊 Creating sample market data...")
    
    # Generate realistic cryptocurrency price data
    np.random.seed(42)  # For reproducible results
    days = 365
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # BTC data
    btc_returns = np.random.normal(0.001, 0.04, days)  # 0.1% mean, 4% daily vol
    btc_price = 100 * np.exp(np.cumsum(btc_returns))
    btc_data = pd.DataFrame({
        'close': btc_price,
        'Close': btc_price,
        'high': btc_price * (1 + np.abs(np.random.normal(0, 0.02, days))),
        'High': btc_price * (1 + np.abs(np.random.normal(0, 0.02, days))),
        'low': btc_price * (1 - np.abs(np.random.normal(0, 0.02, days))),
        'Low': btc_price * (1 - np.abs(np.random.normal(0, 0.02, days))),
        'open': btc_price * (1 + np.random.normal(0, 0.01, days)),
        'Open': btc_price * (1 + np.random.normal(0, 0.01, days)),
        'volume': np.random.randint(1000000, 10000000, days),
        'Volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)
    
    # ETH data (correlated with BTC)
    eth_returns = btc_returns * 0.7 + np.random.normal(0, 0.03, days) * 0.3
    eth_price = 50 * np.exp(np.cumsum(eth_returns))
    eth_data = pd.DataFrame({
        'close': eth_price,
        'Close': eth_price,
        'high': eth_price * (1 + np.abs(np.random.normal(0, 0.025, days))),
        'High': eth_price * (1 + np.abs(np.random.normal(0, 0.025, days))),
        'low': eth_price * (1 - np.abs(np.random.normal(0, 0.025, days))),
        'Low': eth_price * (1 - np.abs(np.random.normal(0, 0.025, days))),
        'open': eth_price * (1 + np.random.normal(0, 0.015, days)),
        'Open': eth_price * (1 + np.random.normal(0, 0.015, days)),
        'volume': np.random.randint(500000, 5000000, days),
        'Volume': np.random.randint(500000, 5000000, days)
    }, index=dates)
    
    market_data = {
        'BTC/USDT': btc_data,
        'ETH/USDT': eth_data
    }
    
    logger.info(f"✅ Created market data for {len(market_data)} symbols")
    logger.info(f"   BTC price range: ${btc_price.min():.2f} - ${btc_price.max():.2f}")
    logger.info(f"   ETH price range: ${eth_price.min():.2f} - ${eth_price.max():.2f}")
    
    return market_data

async def demonstrate_risk_first_architecture():
    """Demonstrate the risk-first architecture implementation"""
    logger.info("🛡️ RISK-FIRST ARCHITECTURE DEMO")
    logger.info("=" * 60)
    
    try:
        # Step 1: Create sample market data
        market_data = create_sample_market_data()
        
        # Step 2: Import and initialize orchestrator
        logger.info("\n🔧 Initializing Risk-First Orchestrator...")
        from risk_first_orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator(initial_capital=100000.0)  # $100K capital
        logger.info("✅ Orchestrator initialized")
        
        # Step 3: Process market data through risk-first pipeline
        logger.info("\n🔄 Processing market data through Risk-First pipeline...")
        results = orchestrator.process_market_data(market_data)
        
        # Step 4: Display results
        logger.info(f"\n🎯 PROCESSING RESULTS:")
        logger.info(f"   Status: {results['system_status']}")
        logger.info(f"   Timestamp: {results['timestamp']}")
        
        if results['system_status'] == 'SUCCESS':
            logger.info(f"   Detected Regime: {results['regime_context']['name']}")
            logger.info(f"   Edge Opportunities: {len(results['edge_opportunities'])}")
            logger.info(f"   Approved Positions: {len(results['optimal_positions'])}")
            logger.info(f"   Capital Deployed: ${results['capital_deployed']:,.2f}")
            
            # Show detailed position information
            if results['optimal_positions']:
                logger.info(f"\n📋 APPROVED POSITIONS:")
                for i, position in enumerate(results['optimal_positions'][:3]):  # Show top 3
                    logger.info(f"   {i+1}. {position['symbol']}:")
                    logger.info(f"      Size: {position['position_size']:.4f}")
                    logger.info(f"      Edge Probability: {position['edge_probability']:.3f}")
                    logger.info(f"      Expected Return: {position['expected_return']:.2%}")
                    logger.info(f"      Risk Metrics: {position['risk_metrics']}")
            
            # Show portfolio decisions
            if results['portfolio_decisions']:
                logger.info(f"\n💰 PORTFOLIO ALLOCATIONS:")
                total_allocation = 0
                for decision in results['portfolio_decisions']:
                    allocation_pct = decision['allocation_weight'] * 100
                    total_allocation += allocation_pct
                    logger.info(f"   {decision['symbol']}: {allocation_pct:.2f}% of portfolio")
                logger.info(f"   Total Allocation: {total_allocation:.2f}%")
        
        logger.info(f"\n{'='*60}")
        logger.info("✅ RISK-FIRST ARCHITECTURE DEMO COMPLETED")
        logger.info("🚀 Key achievements:")
        logger.info("   • Implemented risk-first decision architecture")
        logger.info("   • Risk Engine orchestrates all investment decisions")
        logger.info("   • Professional-grade position sizing and risk management")
        logger.info("   • Integrated regime-aware edge classification")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """Main entry point"""
    print("Chloe AI 0.4 - Risk-First Architecture Demo")
    print("Professional trading AI implementation")
    print()
    
    # Run async demo
    asyncio.run(demonstrate_risk_first_architecture())

if __name__ == "__main__":
    main()