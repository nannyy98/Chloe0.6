"""
Simplified Risk-First Architecture Demo
Shows the core concept without complex dependencies
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_position_size: float = 0.02        # Maximum 2% of capital per position
    kelly_fraction: float = 0.25           # Fraction of Kelly criterion
    max_drawdown_limit: float = 0.15       # Maximum 15% drawdown
    regime_risk_multipliers: Dict[str, float] = None

    def __post_init__(self):
        if self.regime_risk_multipliers is None:
            self.regime_risk_multipliers = {
                'STABLE': 1.0,
                'TRENDING': 1.2,
                'VOLATILE': 0.7,
                'CRISIS': 0.3
            }

class SimpleRiskEngine:
    """Simplified risk engine for demonstration"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_params = RiskParameters()
        logger.info(f"🛡️ Simple Risk Engine initialized with ${initial_capital:,.2f}")

    def calculate_kelly_position_size(self, win_rate: float, win_loss_ratio: float, 
                                    regime: str = 'STABLE') -> float:
        """Calculate position size using Kelly criterion"""
        try:
            # Kelly formula: f* = (p*b - q) / b
            loss_rate = 1 - win_rate
            if win_loss_ratio <= 0:
                win_loss_ratio = 1.0
            
            kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
            fractional_kelly = max(0, kelly_fraction) * self.risk_params.kelly_fraction
            
            # Apply regime adjustment
            regime_multiplier = self.risk_params.regime_risk_multipliers.get(regime, 1.0)
            adjusted_kelly = fractional_kelly * regime_multiplier
            
            # Apply position size limits
            max_allowed = min(
                self.risk_params.max_position_size,
                adjusted_kelly,
                0.02  # Hard cap at 2%
            )
            
            return max(0.0, min(max_allowed, 0.02))
            
        except Exception as e:
            logger.error(f"Kelly calculation failed: {e}")
            return 0.01  # Safe fallback

    def assess_opportunity(self, symbol: str, edge_probability: float, 
                          expected_return: float, regime: str) -> Dict:
        """Assess trading opportunity from risk perspective"""
        
        # Calculate position size
        position_size = self.calculate_kelly_position_size(
            win_rate=edge_probability,
            win_loss_ratio=expected_return / 0.01,  # Assuming 1% typical loss
            regime=regime
        )
        
        # Risk metrics
        risk_metrics = {
            'position_risk': position_size,
            'expected_value': position_size * expected_return * self.current_capital,
            'regime_multiplier': self.risk_params.regime_risk_multipliers.get(regime, 1.0),
            'risk_adjusted_return': expected_return / max(position_size, 0.01)
        }
        
        # Approval decision
        approved = (
            edge_probability >= 0.55 and  # Minimum 55% edge probability
            expected_return >= 0.005 and  # Minimum 0.5% expected return
            position_size > 0 and
            risk_metrics['risk_adjusted_return'] >= 1.5  # Minimum 1.5x risk-adjusted return
        )
        
        return {
            'symbol': symbol,
            'approved': approved,
            'position_size': position_size,
            'position_value': position_size * self.current_capital,
            'edge_probability': edge_probability,
            'expected_return': expected_return,
            'risk_metrics': risk_metrics,
            'regime': regime
        }

class SimpleRegimeDetector:
    """Simple regime detection for demonstration"""
    
    def __init__(self):
        self.regimes = ['STABLE', 'TRENDING', 'VOLATILE', 'CRISIS']
        logger.info("📊 Simple Regime Detector initialized")

    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """Detect current market regime"""
        try:
            returns = price_data['close'].pct_change().dropna()
            
            # Simple volatility-based regime detection
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            if volatility < 0.3:  # Low volatility
                return 'STABLE'
            elif volatility < 0.6:  # Moderate volatility
                # Check for trend
                price_trend = (price_data['close'].iloc[-1] / price_data['close'].iloc[0]) - 1
                if abs(price_trend) > 0.2:  # 20% trend
                    return 'TRENDING'
                else:
                    return 'STABLE'
            elif volatility < 1.0:  # High volatility
                return 'VOLATILE'
            else:  # Very high volatility
                return 'CRISIS'
                
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return 'STABLE'  # Default

class SimpleEdgeClassifier:
    """Simple edge classifier for demonstration"""
    
    def __init__(self):
        logger.info("🎯 Simple Edge Classifier initialized")

    def classify_edges(self, market_data: Dict[str, pd.DataFrame], 
                      regime: str) -> List[Dict]:
        """Classify edge opportunities"""
        opportunities = []
        
        for symbol, data in market_data.items():
            try:
                if len(data) < 30:
                    continue
                    
                # Simple edge detection based on recent momentum
                recent_returns = data['close'].pct_change(5).iloc[-1]  # 5-day momentum
                volatility = data['close'].pct_change().std()
                
                # Calculate edge probability (simplified)
                if regime == 'TRENDING':
                    edge_prob = min(0.9, max(0.1, 0.5 + recent_returns * 10))  # Momentum favors trending
                elif regime == 'VOLATILE':
                    edge_prob = 0.45  # Lower confidence in volatile markets
                else:  # STABLE or CRISIS
                    edge_prob = min(0.8, max(0.2, 0.5 + recent_returns * 5))
                
                # Expected return estimation
                expected_return = abs(recent_returns) * 0.3  # 30% of recent move
                
                if edge_prob >= 0.5:  # Only consider decent edges
                    opportunities.append({
                        'symbol': symbol,
                        'edge_probability': edge_prob,
                        'expected_return': expected_return,
                        'recent_momentum': recent_returns,
                        'volatility': volatility
                    })
                    
            except Exception as e:
                logger.warning(f"Edge classification failed for {symbol}: {e}")
        
        # Sort by edge strength
        opportunities.sort(key=lambda x: x['edge_probability'] * x['expected_return'], reverse=True)
        return opportunities

class RiskFirstOrchestratorDemo:
    """Risk-First Orchestrator Demonstration"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.risk_engine = SimpleRiskEngine(initial_capital)
        self.regime_detector = SimpleRegimeDetector()
        self.edge_classifier = SimpleEdgeClassifier()
        logger.info("🎭 Risk-First Orchestrator Demo initialized")

    def process_market_cycle(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Process complete market cycle with risk-first approach"""
        logger.info("🔄 Starting Risk-First Processing Cycle...")
        
        # Step 1: Regime Detection
        logger.info("   1️⃣ Detecting Market Regime...")
        # Use BTC as market proxy
        btc_data = market_data.get('BTC/USDT', list(market_data.values())[0])
        current_regime = self.regime_detector.detect_regime(btc_data)
        logger.info(f"      Detected regime: {current_regime}")
        
        # Step 2: Edge Classification
        logger.info("   2️⃣ Classifying Edge Opportunities...")
        edge_opportunities = self.edge_classifier.classify_edges(market_data, current_regime)
        logger.info(f"      Found {len(edge_opportunities)} opportunities")
        
        # Step 3: Risk Engine Orchestration (CENTRAL DECISION MAKING)
        logger.info("   3️⃣ Risk Engine Orchestration...")
        approved_positions = []
        total_capital_deployed = 0
        
        for opportunity in edge_opportunities:
            assessment = self.risk_engine.assess_opportunity(
                symbol=opportunity['symbol'],
                edge_probability=opportunity['edge_probability'],
                expected_return=opportunity['expected_return'],
                regime=current_regime
            )
            
            if assessment['approved']:
                approved_positions.append(assessment)
                total_capital_deployed += assessment['position_value']
                logger.info(f"      ✅ Approved: {opportunity['symbol']} "
                          f"(Edge: {opportunity['edge_probability']:.3f}, "
                          f"Size: {assessment['position_size']:.4f})")
            else:
                logger.info(f"      ❌ Rejected: {opportunity['symbol']} "
                          f"(Edge: {opportunity['edge_probability']:.3f})")
        
        # Step 4: Portfolio Summary
        logger.info("   4️⃣ Portfolio Construction...")
        portfolio_summary = {
            'total_positions': len(approved_positions),
            'capital_deployed': total_capital_deployed,
            'capital_deployed_pct': (total_capital_deployed / self.initial_capital) * 100,
            'average_edge_probability': np.mean([p['edge_probability'] for p in approved_positions]) if approved_positions else 0,
            'average_expected_return': np.mean([p['expected_return'] for p in approved_positions]) if approved_positions else 0
        }
        
        results = {
            'timestamp': datetime.now(),
            'regime': current_regime,
            'edge_opportunities': len(edge_opportunities),
            'approved_positions': approved_positions,
            'portfolio_summary': portfolio_summary,
            'system_status': 'SUCCESS'
        }
        
        logger.info("✅ Risk-First Processing Cycle Completed")
        return results

def create_demo_data():
    """Create realistic demo market data"""
    np.random.seed(42)
    days = 252  # One year
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # BTC data with realistic characteristics
    btc_base_returns = np.random.normal(0.0005, 0.035, days)  # 0.05% mean, 3.5% daily vol
    # Add some trending periods
    btc_base_returns[50:100] += 0.01  # Bull run
    btc_base_returns[150:200] -= 0.008  # Bear period
    
    btc_price = 40000 * np.exp(np.cumsum(btc_base_returns))
    btc_data = pd.DataFrame({
        'close': btc_price,
        'Close': btc_price,
        'high': btc_price * (1 + np.abs(np.random.normal(0, 0.02, days))),
        'low': btc_price * (1 - np.abs(np.random.normal(0, 0.02, days))),
        'volume': np.random.randint(10000000000, 50000000000, days)
    }, index=dates)
    
    # ETH data (correlated with BTC but more volatile)
    eth_returns = btc_base_returns * 0.6 + np.random.normal(0, 0.04, days) * 0.4
    eth_price = 2500 * np.exp(np.cumsum(eth_returns))
    eth_data = pd.DataFrame({
        'close': eth_price,
        'Close': eth_price,
        'high': eth_price * (1 + np.abs(np.random.normal(0, 0.025, days))),
        'low': eth_price * (1 - np.abs(np.random.normal(0, 0.025, days))),
        'volume': np.random.randint(5000000000, 20000000000, days)
    }, index=dates)
    
    # SOL data (more volatile altcoin)
    sol_returns = btc_base_returns * 0.4 + np.random.normal(0, 0.06, days) * 0.6
    sol_price = 100 * np.exp(np.cumsum(sol_returns))
    sol_data = pd.DataFrame({
        'close': sol_price,
        'Close': sol_price,
        'high': sol_price * (1 + np.abs(np.random.normal(0, 0.04, days))),
        'low': sol_price * (1 - np.abs(np.random.normal(0, 0.04, days))),
        'volume': np.random.randint(1000000000, 5000000000, days)
    }, index=dates)
    
    return {
        'BTC/USDT': btc_data,
        'ETH/USDT': eth_data,
        'SOL/USDT': sol_data
    }

def main():
    """Main demonstration"""
    print("🎭 CHLOE AI 0.4 - RISK-FIRST ARCHITECTURE DEMO")
    print("=" * 60)
    print("Professional Trading AI Implementation")
    print()
    
    try:
        # Create demo data
        market_data = create_demo_data()
        print(f"📊 Market Data Created:")
        for symbol, data in market_data.items():
            print(f"   {symbol}: ${data['close'].iloc[-1]:.2f} (Range: ${data['close'].min():.2f} - ${data['close'].max():.2f})")
        print()
        
        # Initialize orchestrator
        orchestrator = RiskFirstOrchestratorDemo(initial_capital=100000.0)
        print("✅ Risk-First Orchestrator Initialized")
        print()
        
        # Process market cycle
        results = orchestrator.process_market_cycle(market_data)
        
        # Display results
        print("🎯 PROCESSING RESULTS:")
        print(f"   Status: {results['system_status']}")
        print(f"   Timestamp: {results['timestamp']}")
        print(f"   Market Regime: {results['regime']}")
        print(f"   Edge Opportunities: {results['edge_opportunities']}")
        print(f"   Approved Positions: {len(results['approved_positions'])}")
        print()
        
        if results['approved_positions']:
            print("📋 APPROVED POSITIONS:")
            for i, position in enumerate(results['approved_positions']):
                print(f"   {i+1}. {position['symbol']}:")
                print(f"      Edge Probability: {position['edge_probability']:.3f}")
                print(f"      Expected Return: {position['expected_return']:.2%}")
                print(f"      Position Size: {position['position_size']:.4f} (${position['position_value']:,.2f})")
                print(f"      Risk Metrics: {position['risk_metrics']}")
                print()
        
        portfolio = results['portfolio_summary']
        print("💰 PORTFOLIO SUMMARY:")
        print(f"   Total Positions: {portfolio['total_positions']}")
        print(f"   Capital Deployed: ${portfolio['capital_deployed']:,.2f} ({portfolio['capital_deployed_pct']:.2f}%)")
        print(f"   Average Edge Probability: {portfolio['average_edge_probability']:.3f}")
        print(f"   Average Expected Return: {portfolio['average_expected_return']:.2%}")
        print()
        
        print("=" * 60)
        print("✅ RISK-FIRST ARCHITECTURE DEMO COMPLETED")
        print("🚀 Key Achievements:")
        print("   • Implemented risk-first decision architecture")
        print("   • Risk Engine orchestrates all investment decisions")
        print("   • Professional-grade position sizing and risk management")
        print("   • Integrated regime-aware edge classification")
        print("   • Demonstrated complete processing pipeline")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()