"""
Risk-First Orchestrator for Chloe AI 0.4
Implements professional trading AI architecture where Risk Engine orchestrates all decisions
Based on Aziz Salimov's industry recommendations
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Import core components
from regime_detection import RegimeDetector, get_regime_detector
from enhanced_risk_engine import EnhancedRiskEngine, get_enhanced_risk_engine
from edge_classifier import EdgeClassifier, get_edge_classifier
from portfolio_constructor import PortfolioConstructor, get_portfolio_constructor
from feature_store.feature_calculator import FeatureCalculator, get_feature_calculator

logger = logging.getLogger(__name__)

class RiskFirstOrchestrator:
    """
    Main orchestrator that puts Risk Engine at the center of decision-making
    Professional architecture: Market → Features → Regime → Edge → Risk → Portfolio → Execution
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Initialize core components
        self._initialize_components()
        
        logger.info("🛡️ Risk-First Orchestrator initialized")
        logger.info(f"   Initial Capital: ${self.initial_capital:,.2f}")
        
    def _initialize_components(self):
        """Initialize all system components with proper integration"""
        try:
            # Get singleton instances
            self.regime_detector = get_regime_detector()
            self.risk_engine = get_enhanced_risk_engine(self.initial_capital)
            self.edge_classifier = get_edge_classifier('ensemble')
            self.portfolio_constructor = get_portfolio_constructor(self.initial_capital)
            self.feature_calculator = get_feature_calculator()
            
            # Initialize portfolio tracking
            self.risk_engine.initialize_portfolio_tracking(self.initial_capital)
            self.portfolio_constructor.initialize_portfolio()
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def process_market_data(self, market_data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Main processing pipeline - Risk Engine orchestrates everything
        
        Args:
            market_data_dict: Dictionary of symbol -> DataFrame with market data
            
        Returns:
            Processing results with decisions and risk assessments
        """
        try:
            logger.info("🔄 Starting Risk-First processing cycle...")
            
            # Step 1: Feature Engineering (Risk Engine guides what features to calculate)
            logger.info("   1️⃣ Feature Engineering...")
            enhanced_features = self._enhanced_feature_processing(market_data_dict)
            
            # Step 2: Regime Detection (Risk Engine needs market context)
            logger.info("   2️⃣ Regime Detection...")
            regime_context = self._integrated_regime_detection(enhanced_features)
            
            # Step 3: Edge Classification (Risk Engine evaluates edge probabilities)
            logger.info("   3️⃣ Edge Classification...")
            edge_opportunities = self._risk_guided_edge_classification(enhanced_features, regime_context)
            
            # Step 4: Risk Engine Decision Making (Central orchestration)
            logger.info("   4️⃣ Risk Engine Orchestration...")
            optimal_positions = self._risk_engine_orchestration(edge_opportunities, regime_context)
            
            # Step 5: Portfolio Construction (Risk Engine approved allocations)
            logger.info("   5️⃣ Portfolio Construction...")
            portfolio_decisions = self._portfolio_execution(optimal_positions)
            
            # Compile results
            results = {
                'timestamp': datetime.now(),
                'regime_context': regime_context,
                'edge_opportunities': edge_opportunities,
                'optimal_positions': optimal_positions,
                'portfolio_decisions': portfolio_decisions,
                'system_status': 'SUCCESS',
                'capital_deployed': sum(abs(pos.get('position_value', 0)) for pos in portfolio_decisions)
            }
            
            logger.info(f"✅ Processing cycle completed. Capital deployed: ${results['capital_deployed']:,.2f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Processing cycle failed: {e}")
            return {
                'timestamp': datetime.now(),
                'system_status': 'ERROR',
                'error_message': str(e)
            }
    
    def _enhanced_feature_processing(self, market_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Enhanced feature engineering guided by risk considerations
        """
        enhanced_data = {}
        
        for symbol, df in market_data_dict.items():
            try:
                # Calculate base features
                features_df = self.feature_calculator.calculate_all_features(df, symbol=symbol)
                
                # Add risk-sensitive features
                risk_features = self._calculate_risk_sensitive_features(df, symbol)
                for col, values in risk_features.items():
                    features_df[col] = values
                    
                enhanced_data[symbol] = features_df
                
            except Exception as e:
                logger.warning(f"⚠️ Feature processing failed for {symbol}: {e}")
                enhanced_data[symbol] = df  # Fallback to original data
                
        return enhanced_data
    
    def _calculate_risk_sensitive_features(self, df: pd.DataFrame, symbol: str) -> Dict[str, pd.Series]:
        """
        Calculate features specifically designed for risk management
        """
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        returns = close_prices.pct_change().dropna()
        
        risk_features = {}
        
        # Volatility regime features
        rolling_vol = returns.rolling(20).std()
        risk_features['volatility_regime'] = rolling_vol / rolling_vol.rolling(200).std()
        risk_features['volatility_spike'] = (rolling_vol > rolling_vol.rolling(20).mean() * 2).astype(int)
        
        # Drawdown features
        rolling_max = close_prices.rolling(252).max()
        risk_features['current_drawdown'] = (close_prices - rolling_max) / rolling_max
        
        # Liquidity risk features
        volume = df['volume'] if 'volume' in df.columns else df['Volume']
        risk_features['volume_stress'] = (volume.rolling(20).mean() / volume.rolling(200).mean()) < 0.5
        
        # Correlation risk features (simplified)
        risk_features['correlation_stress'] = abs(returns.autocorr(lag=1)) > 0.3
        
        return risk_features
    
    def _integrated_regime_detection(self, enhanced_features: Dict[str, pd.DataFrame]) -> Dict:
        """
        Integrated regime detection using enhanced features
        """
        # Use BTC as market proxy for regime detection
        btc_data = enhanced_features.get('BTC/USDT', list(enhanced_features.values())[0])
        
        try:
            regime_result = self.regime_detector.detect_current_regime(btc_data[['close']])
            
            regime_context = {
                'name': regime_result.name if regime_result else 'STABLE',
                'probability': regime_result.probability if regime_result else 0.5,
                'confidence': regime_result.confidence if regime_result else 0.5,
                'features': regime_result.features if regime_result else {}
            }
            
            logger.info(f"      Detected regime: {regime_context['name']} (conf: {regime_context['confidence']:.2f})")
            return regime_context
            
        except Exception as e:
            logger.warning(f"⚠️ Regime detection failed: {e}")
            return {
                'name': 'STABLE',
                'probability': 0.5,
                'confidence': 0.5,
                'features': {}
            }
    
    def _risk_guided_edge_classification(self, enhanced_features: Dict[str, pd.DataFrame], 
                                       regime_context: Dict) -> List[Dict]:
        """
        Edge classification guided by risk engine requirements
        """
        opportunities = []
        
        for symbol, features_df in enhanced_features.items():
            try:
                if len(features_df) < 50:  # Need sufficient data
                    continue
                
                # Get latest data for edge assessment
                recent_data = features_df.tail(100)
                
                # Prepare features for edge classification
                edge_features = self.edge_classifier.prepare_edge_features(recent_data, regime_context)
                
                if edge_features is not None:
                    # Get edge probability
                    edge_result = self.edge_classifier.predict_edge_opportunity(edge_features)
                    
                    if edge_result and edge_result.has_edge:
                        opportunity = {
                            'symbol': symbol,
                            'edge_probability': edge_result.edge_probability,
                            'edge_strength': edge_result.edge_strength,
                            'expected_return': edge_result.expected_return,
                            'holding_period': edge_result.holding_period,
                            'stop_loss': edge_result.stop_loss_level,
                            'take_profit': edge_result.take_profit_level,
                            'regime_context': regime_context
                        }
                        opportunities.append(opportunity)
                        
            except Exception as e:
                logger.warning(f"⚠️ Edge classification failed for {symbol}: {e}")
        
        # Sort by edge strength
        opportunities.sort(key=lambda x: x['edge_strength'], reverse=True)
        logger.info(f"      Found {len(opportunities)} edge opportunities")
        
        return opportunities
    
    def _risk_engine_orchestration(self, edge_opportunities: List[Dict], 
                                 regime_context: Dict) -> List[Dict]:
        """
        Risk Engine makes final decisions on position sizing and approval
        This is the CENTRAL ORCHESTRATION point
        """
        approved_positions = []
        
        for opportunity in edge_opportunities:
            try:
                # Risk Engine evaluates this opportunity
                risk_assessment = self.risk_engine.assess_position_risk(
                    symbol=opportunity['symbol'],
                    entry_price=opportunity['take_profit'],  # Will be adjusted
                    position_size=opportunity['edge_strength'] * self.current_capital * 0.1,  # Base sizing
                    volatility=opportunity.get('volatility', 0.02),
                    regime=regime_context['name']
                )
                
                if risk_assessment['approved']:
                    # Risk Engine approves - calculate final position
                    final_position = self._calculate_final_position(opportunity, risk_assessment, regime_context)
                    approved_positions.append(final_position)
                    
                    logger.info(f"      ✅ Approved: {opportunity['symbol']} "
                              f"(Edge: {opportunity['edge_probability']:.3f}, "
                              f"Size: {final_position['position_size']:.4f})")
                else:
                    logger.info(f"      ❌ Rejected: {opportunity['symbol']} - "
                              f"{risk_assessment['rejection_reason']}")
                    
            except Exception as e:
                logger.error(f"❌ Risk assessment failed for {opportunity['symbol']}: {e}")
        
        logger.info(f"      Approved {len(approved_positions)} positions out of {len(edge_opportunities)} opportunities")
        return approved_positions
    
    def _calculate_final_position(self, opportunity: Dict, risk_assessment: Dict, 
                                regime_context: Dict) -> Dict:
        """
        Calculate final position parameters incorporating all risk factors
        """
        # Base position from edge strength
        base_size = opportunity['edge_strength'] * self.current_capital * 0.05  # 5% base
        
        # Apply Kelly criterion from risk engine
        kelly_size = self.risk_engine.calculate_kelly_position_size(
            win_rate=opportunity['edge_probability'],
            win_loss_ratio=2.0,  # Assumed 2:1 reward:risk
            account_size=self.current_capital,
            regime=regime_context['name']
        )
        
        # Use conservative sizing
        final_size = min(base_size, kelly_size * self.current_capital, 
                        self.current_capital * 0.02)  # Max 2% of capital
        
        # Adjust for regime
        regime_multiplier = self.risk_engine.risk_params.regime_risk_multiplier.get(
            regime_context['name'], 1.0
        )
        final_size *= regime_multiplier
        
        return {
            'symbol': opportunity['symbol'],
            'position_size': final_size,
            'position_value': final_size,
            'entry_price': opportunity['take_profit'],  # Simplified
            'stop_loss': opportunity['stop_loss'],
            'take_profit': opportunity['take_profit'],
            'edge_probability': opportunity['edge_probability'],
            'risk_metrics': risk_assessment['risk_metrics'],
            'regime_context': regime_context
        }
    
    def _portfolio_execution(self, approved_positions: List[Dict]) -> List[Dict]:
        """
        Execute portfolio construction with risk-approved positions
        """
        try:
            # Convert to portfolio constructor format
            portfolio_input = {}
            for pos in approved_positions:
                portfolio_input[pos['symbol']] = {
                    'weight': pos['position_size'] / self.current_capital,
                    'expected_return': pos['edge_probability'] * 0.02,  # Simplified
                    'volatility': 0.02,  # Default
                    'correlation': 0.1    # Default low correlation
                }
            
            # Construct optimal portfolio
            allocations = self.portfolio_constructor.construct_optimal_portfolio(
                market_data_dict={},  # Would be actual data in real system
                regime_context=None   # Would be actual regime in real system
            )
            
            # Convert back to decision format
            decisions = []
            for allocation in allocations:
                decisions.append({
                    'symbol': allocation.symbol,
                    'position_size': allocation.position_size,
                    'position_value': allocation.position_size * allocation.entry_price,
                    'allocation_weight': allocation.weight,
                    'edge_probability': allocation.edge_probability
                })
            
            return decisions
            
        except Exception as e:
            logger.error(f"❌ Portfolio execution failed: {e}")
            return []

# Global orchestrator instance
orchestrator = None

def get_orchestrator(initial_capital: float = 10000.0) -> RiskFirstOrchestrator:
    """Get singleton orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = RiskFirstOrchestrator(initial_capital)
    return orchestrator

def main():
    """Example usage"""
    print("Risk-First Orchestrator ready for professional trading AI architecture")

if __name__ == "__main__":
    main()