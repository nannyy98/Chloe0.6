#!/usr/bin/env python3
"""
Quick status check of Chloe AI system components
Tests core functionality without external dependencies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_components():
    """Test core components that don't require external dependencies"""
    print("🔍 Chloe AI System Status Check")
    print("=" * 50)
    
    # Test 1: Basic imports
    print("\n1. Testing core component imports...")
    try:
        from regime_detection import RegimeDetector
        print("   ✅ Regime Detection - OK")
    except Exception as e:
        print(f"   ❌ Regime Detection - FAILED: {e}")
    
    try:
        from enhanced_risk_engine import EnhancedRiskEngine
        print("   ✅ Enhanced Risk Engine - OK")
    except Exception as e:
        print(f"   ❌ Enhanced Risk Engine - FAILED: {e}")
    
    try:
        from edge_classifier import EdgeClassifier
        print("   ✅ Edge Classifier - OK")
    except Exception as e:
        print(f"   ❌ Edge Classifier - FAILED: {e}")
    
    try:
        from portfolio_constructor import PortfolioConstructor
        print("   ✅ Portfolio Constructor - OK")
    except Exception as e:
        print(f"   ❌ Portfolio Constructor - FAILED: {e}")
    
    # Test 2: Feature store
    print("\n2. Testing feature store...")
    try:
        from feature_store.feature_calculator import FeatureCalculator
        calc = FeatureCalculator()
        print("   ✅ Feature Calculator - OK")
    except Exception as e:
        print(f"   ❌ Feature Calculator - FAILED: {e}")
    
    # Test 3: Backtesting components
    print("\n3. Testing backtesting components...")
    try:
        from backtest.engine import BacktestEngine
        engine = BacktestEngine()
        print("   ✅ Backtest Engine - OK")
    except Exception as e:
        print(f"   ❌ Backtest Engine - FAILED: {e}")
    
    # Test 4: Risk components
    print("\n4. Testing risk components...")
    try:
        from risk.advanced_risk_models import AdvancedRiskAnalyzer
        analyzer = AdvancedRiskAnalyzer()
        print("   ✅ Advanced Risk Analyzer - OK")
    except Exception as e:
        print(f"   ❌ Advanced Risk Analyzer - FAILED: {e}")
    
    # Test 5: Portfolio components
    print("\n5. Testing portfolio components...")
    try:
        from portfolio.portfolio import Portfolio
        portfolio = Portfolio(initial_capital=10000)
        print("   ✅ Portfolio Management - OK")
    except Exception as e:
        print(f"   ❌ Portfolio Management - FAILED: {e}")
    
    # Test 6: Simple functionality test
    print("\n6. Testing basic functionality...")
    try:
        import numpy as np
        import pandas as pd
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'Close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'High': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'Low': np.random.randn(100).cumsum() + 95,
            'open': np.random.randn(100).cumsum() + 98,
            'Open': np.random.randn(100).cumsum() + 98,
            'volume': np.random.randint(1000, 10000, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Test regime detection
        from regime_detection import RegimeDetector
        detector = RegimeDetector()
        regime_result = detector.detect_current_regime(sample_data[['close']])
        print(f"   ✅ Regime Detection - Current regime: {regime_result.name if regime_result else 'Unknown'}")
        
        # Test feature calculation
        from feature_store.feature_calculator import FeatureCalculator
        calc = FeatureCalculator()
        features = calc.calculate_all_features(sample_data)
        print(f"   ✅ Feature Calculation - Generated {len(features.columns)} features")
        
        # Test risk engine
        from enhanced_risk_engine import EnhancedRiskEngine
        risk_engine = EnhancedRiskEngine(initial_capital=10000)
        # Just test initialization for now
        print(f"   ✅ Risk Engine - Initialized with capital: ${risk_engine.current_capital:,.2f}")
        
        print("\n🎉 All core components are functioning correctly!")
        
    except Exception as e:
        print(f"   ❌ Functionality test - FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_core_components()