"""
Portfolio Construction Logic for Chloe AI 0.4
Implements professional portfolio optimization combining regime detection, edge classification, and risk management
Creates optimal capital allocation decisions based on edge probabilities and risk constraints
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from enhanced_risk_engine import EnhancedRiskEngine, get_enhanced_risk_engine
from edge_classifier import EdgeClassifier, get_edge_classifier
from regime_detection import RegimeDetector

logger = logging.getLogger(__name__)

@dataclass
class PortfolioAllocation:
    """Represents a portfolio allocation decision"""
    symbol: str
    weight: float                    # Portfolio weight (0-1)
    position_size: float             # Actual position size in units
    entry_price: float               # Entry price
    stop_loss: float                 # Stop loss level
    take_profit: float               # Take profit level
    edge_probability: float          # Edge classifier confidence
    risk_adjusted_return: float      # Risk-adjusted expected return
    regime_context: str              # Market regime during allocation
    allocation_timestamp: datetime
    volatility: float = 0.0          # Asset volatility (added for portfolio calculations)

@dataclass
class PortfolioConstraints:
    """Portfolio-level constraints"""
    max_positions: int = 10          # Maximum number of concurrent positions
    max_sector_exposure: float = 0.30  # Maximum exposure to any sector/group
    max_correlation: float = 0.7     # Maximum correlation between positions
    minimum_edge_threshold: float = 0.6  # Minimum edge probability for inclusion
    max_portfolio_volatility: float = 0.15  # Maximum portfolio volatility target
    diversification_penalty: float = 0.1   # Penalty for concentration

class PortfolioConstructor:
    """
    Professional portfolio construction that optimizes capital allocation
    Based on edge probabilities, risk constraints, and market regime context
    """
    
    def __init__(self, initial_capital: float = 10000.0, 
                 constraints: PortfolioConstraints = None):
        self.initial_capital = initial_capital
        self.constraints = constraints or PortfolioConstraints()
        self.current_portfolio = {}  # symbol -> PortfolioAllocation
        self.allocation_history = []
        self.is_initialized = False
        
        # Component integrations
        self.risk_engine = get_enhanced_risk_engine(initial_capital)
        self.edge_classifier = get_edge_classifier('ensemble')
        self.regime_detector = RegimeDetector(n_regimes=4)
        
        logger.info("📊 Portfolio Constructor initialized")
        logger.info(f"   Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"   Max positions: {self.constraints.max_positions}")
        logger.info(f"   Min edge threshold: {self.constraints.minimum_edge_threshold}")
    
    def initialize_portfolio(self) -> bool:
        """Initialize portfolio tracking"""
        try:
            self.risk_engine.initialize_portfolio_tracking(self.initial_capital)
            self.is_initialized = True
            logger.info("✅ Portfolio constructor initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Portfolio initialization failed: {e}")
            return False
    
    def construct_optimal_portfolio(self, 
                                  market_data_dict: Dict[str, pd.DataFrame],
                                  regime_context: Dict = None,
                                  existing_positions: Dict = None) -> List[PortfolioAllocation]:
        """
        Construct optimal portfolio allocation based on edge probabilities and constraints
        
        Args:
            market_data_dict: Dictionary mapping symbols to market data DataFrames
            regime_context: Current market regime information
            existing_positions: Current portfolio positions
            
        Returns:
            List of optimal portfolio allocations
        """
        try:
            logger.info("🏗️ Constructing optimal portfolio...")
            
            if not self.is_initialized:
                self.initialize_portfolio()
            
            # Step 1: Evaluate edge opportunities for all symbols
            edge_opportunities = self._evaluate_edge_opportunities(
                market_data_dict, regime_context
            )
            
            logger.info(f"   Found {len(edge_opportunities)} edge opportunities")
            if edge_opportunities:
                logger.info(f"   Best edge probability: {max([e['edge_probability'] for e in edge_opportunities]):.3f}")
            
            # Step 2: Filter by minimum edge threshold
            qualified_opportunities = [
                opp for opp in edge_opportunities 
                if opp['edge_probability'] >= self.constraints.minimum_edge_threshold
            ]
            
            logger.info(f"   Qualified opportunities: {len(qualified_opportunities)} "
                       f"(threshold: {self.constraints.minimum_edge_threshold})")
            
            if not qualified_opportunities:
                logger.info("   ⚠️ No qualified opportunities meeting edge threshold")
                return []
            
            # Step 3: Rank opportunities by risk-adjusted edge score
            ranked_opportunities = self._rank_opportunities(qualified_opportunities)
            
            # Step 4: Optimize portfolio allocation with constraints
            optimal_allocations = self._optimize_portfolio_allocation(
                ranked_opportunities, existing_positions
            )
            
            # Step 5: Validate allocations through risk engine
            validated_allocations = self._validate_allocations(optimal_allocations)
            
            # Step 6: Record allocation decision
            self._record_allocation(validated_allocations)
            
            logger.info(f"✅ Portfolio construction completed with {len(validated_allocations)} positions")
            return validated_allocations
            
        except Exception as e:
            logger.error(f"❌ Portfolio construction failed: {e}")
            return []
    
    def _evaluate_edge_opportunities(self, 
                                   market_data_dict: Dict[str, pd.DataFrame],
                                   regime_context: Dict = None) -> List[Dict]:
        """Evaluate edge opportunities for all available symbols"""
        opportunities = []
        
        for symbol, market_data in market_data_dict.items():
            try:
                if len(market_data) < 50:  # Need sufficient data
                    continue
                
                # Get latest data for edge assessment
                recent_data = market_data.tail(100)
                
                # Prepare features for edge classification
                features = self.edge_classifier.prepare_edge_features(
                    market_data=recent_data,
                    regime_info=regime_context
                )
                
                if len(features) == 0:
                    continue
                
                # Get edge prediction
                predictions = self.edge_classifier.predict_edge(features.tail(10))
                if len(predictions) == 0:
                    continue
                
                edge_probability = predictions['ensemble_prob'].iloc[-1]
                
                # Calculate basic position parameters
                current_price = market_data['close'].iloc[-1]
                volatility = market_data['close'].pct_change().tail(20).std()
                
                # Simple stop loss and take profit levels
                stop_loss = current_price * (1 - 0.02)  # 2% stop loss
                take_profit = current_price * (1 + 0.04)  # 4% take profit
                
                opportunity = {
                    'symbol': symbol,
                    'edge_probability': float(edge_probability),
                    'current_price': float(current_price),
                    'volatility': float(volatility),
                    'stop_loss': float(stop_loss),
                    'take_profit': float(take_profit),
                    'regime_context': regime_context.get('name', 'UNKNOWN') if regime_context else 'UNKNOWN'
                }
                
                opportunities.append(opportunity)
                
            except Exception as e:
                logger.debug(f"Failed to evaluate {symbol}: {e}")
                continue
        
        return opportunities
    
    def _rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Rank opportunities by composite edge score"""
        for opp in opportunities:
            # Composite score combining edge probability and risk metrics
            edge_component = opp['edge_probability']
            
            # Risk adjustment (lower volatility = better)
            risk_component = max(0, 1 - opp['volatility'] / 0.05)  # Normalize volatility
            
            # Regime adjustment
            regime_multiplier = self._get_regime_multiplier(opp['regime_context'])
            
            # Final composite score
            opp['composite_score'] = edge_component * risk_component * regime_multiplier
            opp['risk_adjusted_return'] = edge_component * 0.02  # Simplified expected return
        
        # Sort by composite score (descending)
        ranked = sorted(opportunities, key=lambda x: x['composite_score'], reverse=True)
        return ranked
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """Get regime-specific opportunity multiplier"""
        multipliers = {
            'TRENDING': 1.2,      # Better for momentum strategies
            'MEAN_REVERTING': 1.1, # Good for mean-reversion
            'STABLE': 0.8,        # Moderate opportunities
            'VOLATILE': 0.9       # Risky but potentially rewarding
        }
        return multipliers.get(regime, 1.0)
    
    def _optimize_portfolio_allocation(self, 
                                     opportunities: List[Dict],
                                     existing_positions: Dict = None) -> List[PortfolioAllocation]:
        """Optimize portfolio allocation using constrained optimization"""
        if not opportunities:
            return []
        
        # Limit to maximum positions
        top_opportunities = opportunities[:self.constraints.max_positions]
        
        # Calculate initial weights using edge probabilities
        total_edge_score = sum(opp['composite_score'] for opp in top_opportunities)
        if total_edge_score == 0:
            return []
        
        allocations = []
        remaining_capital = self.initial_capital
        
        for opp in top_opportunities:
            # Weight proportional to edge score
            weight = opp['composite_score'] / total_edge_score
            
            # Apply position sizing through risk engine
            position_value = weight * remaining_capital
            
            # Get risk assessment
            risk_assessment = self.risk_engine.assess_position_risk(
                symbol=opp['symbol'],
                entry_price=opp['current_price'],
                position_size=position_value / opp['current_price'],
                stop_loss=opp['stop_loss'],
                take_profit=opp['take_profit'],
                volatility=opp['volatility'],
                regime=opp['regime_context']
            )
            
            if risk_assessment['approved']:
                # Create allocation
                allocation = PortfolioAllocation(
                    symbol=opp['symbol'],
                    weight=weight,
                    position_size=position_value / opp['current_price'],
                    entry_price=opp['current_price'],
                    stop_loss=opp['stop_loss'],
                    take_profit=opp['take_profit'],
                    edge_probability=opp['edge_probability'],
                    risk_adjusted_return=opp['risk_adjusted_return'],
                    regime_context=opp['regime_context'],
                    allocation_timestamp=datetime.now(),
                    volatility=opp['volatility']
                )
                
                allocations.append(allocation)
                remaining_capital -= position_value
                
                logger.info(f"   ✅ Allocated {weight*100:.1f}% to {opp['symbol']} "
                           f"(Edge: {opp['edge_probability']:.3f})")
            else:
                logger.info(f"   ❌ Rejected {opp['symbol']} due to risk constraints")
        
        return allocations
    
    def _validate_allocations(self, allocations: List[PortfolioAllocation]) -> List[PortfolioAllocation]:
        """Validate final portfolio allocations against all constraints"""
        if not allocations:
            return []
        
        # Check portfolio-level constraints
        total_weight = sum(alloc.weight for alloc in allocations)
        
        # Normalize weights to sum to 1
        if total_weight > 0:
            for alloc in allocations:
                alloc.weight = alloc.weight / total_weight
        
        # Portfolio volatility check
        portfolio_vol = self._calculate_portfolio_volatility(allocations)
        if portfolio_vol > self.constraints.max_portfolio_volatility:
            logger.warning(f"⚠️ Portfolio volatility {portfolio_vol:.3f} exceeds limit "
                          f"{self.constraints.max_portfolio_volatility:.3f}")
            # Reduce position sizes proportionally
            reduction_factor = self.constraints.max_portfolio_volatility / portfolio_vol
            for alloc in allocations:
                alloc.weight *= reduction_factor
                alloc.position_size *= reduction_factor
        
        logger.info(f"   Final portfolio: {len(allocations)} positions, "
                   f"volatility: {portfolio_vol:.3f}")
        
        return allocations
    
    def _calculate_portfolio_volatility(self, allocations: List[PortfolioAllocation]) -> float:
        """Calculate estimated portfolio volatility"""
        if not allocations:
            return 0.0
        
        # Simplified: weighted average of individual volatilities
        total_weight = sum(alloc.weight for alloc in allocations)
        if total_weight == 0:
            return 0.0
        
        weighted_vol = sum(alloc.weight * alloc.volatility for alloc in allocations)
        return weighted_vol / total_weight
    
    def _record_allocation(self, allocations: List[PortfolioAllocation]):
        """Record allocation decision for tracking"""
        allocation_record = {
            'timestamp': datetime.now(),
            'allocations': allocations,
            'total_positions': len(allocations),
            'total_weight': sum(alloc.weight for alloc in allocations),
            'portfolio_value': sum(alloc.position_size * alloc.entry_price for alloc in allocations)
        }
        
        self.allocation_history.append(allocation_record)
        self.current_portfolio = {alloc.symbol: alloc for alloc in allocations}
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        if not self.current_portfolio:
            return {
                'status': 'EMPTY',
                'positions': 0,
                'total_value': 0.0,
                'cash': self.initial_capital,
                'leverage': 0.0,
                'allocation_history': len(self.allocation_history)
            }
        
        total_value = sum(
            alloc.position_size * alloc.entry_price 
            for alloc in self.current_portfolio.values()
        )
        
        cash = self.initial_capital - total_value
        
        return {
            'status': 'ACTIVE',
            'positions': len(self.current_portfolio),
            'total_value': total_value,
            'cash': cash,
            'leverage': total_value / self.initial_capital if self.initial_capital > 0 else 0,
            'allocation_history': len(self.allocation_history)
        }
    
    def rebalance_portfolio(self, 
                          market_data_dict: Dict[str, pd.DataFrame],
                          regime_context: Dict = None) -> List[PortfolioAllocation]:
        """Rebalance existing portfolio based on new information"""
        logger.info("🔄 Rebalancing portfolio...")
        
        # Get current positions
        existing_positions = {symbol: alloc for symbol, alloc in self.current_portfolio.items()}
        
        # Construct new optimal portfolio
        new_allocations = self.construct_optimal_portfolio(
            market_data_dict, regime_context, existing_positions
        )
        
        # Calculate changes needed
        changes = self._calculate_rebalancing_changes(existing_positions, new_allocations)
        
        logger.info(f"   Rebalancing complete: {len(changes)} changes")
        return new_allocations
    
    def _calculate_rebalancing_changes(self, 
                                     current: Dict[str, PortfolioAllocation],
                                     new: List[PortfolioAllocation]) -> List[Dict]:
        """Calculate specific rebalancing actions needed"""
        changes = []
        new_symbols = {alloc.symbol for alloc in new}
        current_symbols = set(current.keys())
        
        # Positions to close (no longer in new allocation)
        for symbol in current_symbols - new_symbols:
            changes.append({
                'action': 'CLOSE',
                'symbol': symbol,
                'current_weight': current[symbol].weight,
                'reason': 'No longer meets criteria'
            })
        
        # Positions to adjust
        for alloc in new:
            if alloc.symbol in current:
                weight_change = alloc.weight - current[alloc.symbol].weight
                if abs(weight_change) > 0.01:  # 1% threshold
                    changes.append({
                        'action': 'ADJUST',
                        'symbol': alloc.symbol,
                        'current_weight': current[alloc.symbol].weight,
                        'new_weight': alloc.weight,
                        'change': weight_change
                    })
        
        # New positions to open
        for symbol in new_symbols - current_symbols:
            new_alloc = next(a for a in new if a.symbol == symbol)
            changes.append({
                'action': 'OPEN',
                'symbol': symbol,
                'weight': new_alloc.weight,
                'reason': 'New opportunity identified'
            })
        
        return changes

# Global portfolio constructor instance
portfolio_constructor = None

def get_portfolio_constructor(initial_capital: float = 10000.0) -> PortfolioConstructor:
    """Get singleton portfolio constructor instance"""
    global portfolio_constructor
    if portfolio_constructor is None:
        portfolio_constructor = PortfolioConstructor(initial_capital)
    return portfolio_constructor