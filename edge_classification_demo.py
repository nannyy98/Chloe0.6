"""
Edge Classification Demo for Chloe AI 0.4
Demonstrates edge detection focused on identifying genuine trading opportunities
Rather than predicting prices, this identifies probability of having a statistical edge
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from edge_classifier import EdgeClassifier, get_edge_classifier
from regime_detection import RegimeDetector
from enhanced_risk_engine import EnhancedRiskEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_edge_classification():
    """Demonstrate edge classification capabilities"""
    logger.info("🎯 Chloe AI 0.4 - Edge Classification Demo")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        logger.info("🔧 Initializing components...")
        
        # Edge classifier
        edge_clf = get_edge_classifier('ensemble')
        
        # Regime detector
        regime_detector = RegimeDetector(n_regimes=4)
        
        # Risk engine
        risk_engine = EnhancedRiskEngine(initial_capital=10000.0)
        risk_engine.initialize_portfolio_tracking()
        
        logger.info("✅ All components initialized")
        
        # Generate synthetic market data for testing
        logger.info("\n📊 Generating synthetic market data...")
        
        # Create realistic price series with different market conditions
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        # Simulate different market regimes
        n_points = len(dates)
        
        # Base price series with regime changes
        prices = []
        current_price = 50000  # Starting BTC-like price
        
        # Regime durations (in days)
        regime_changes = [(0, 'STABLE'), (90, 'TRENDING'), (180, 'VOLATILE'), (270, 'MEAN_REVERTING')]
        current_regime_idx = 0
        
        for i in range(n_points):
            # Update regime
            if current_regime_idx < len(regime_changes) - 1 and i >= regime_changes[current_regime_idx + 1][0]:
                current_regime_idx += 1
            
            regime = regime_changes[current_regime_idx][1]
            
            # Different dynamics for each regime
            if regime == 'TRENDING':
                # Strong trend with momentum
                drift = 0.001 + np.random.normal(0, 0.005)
                volatility = 0.02
            elif regime == 'VOLATILE':
                # High volatility, no clear direction
                drift = np.random.normal(0, 0.002)
                volatility = 0.04
            elif regime == 'MEAN_REVERTING':
                # Mean-reverting behavior
                target_price = 50000
                drift = (target_price - current_price) * 0.0001 + np.random.normal(0, 0.001)
                volatility = 0.015
            else:  # STABLE
                # Low volatility, slight drift
                drift = np.random.normal(0.0001, 0.002)
                volatility = 0.01
            
            # Price update
            price_change = drift + np.random.normal(0, volatility)
            current_price = current_price * (1 + price_change)
            current_price = max(current_price, 1000)  # Floor price
            prices.append(current_price)
        
        # Create market data DataFrame
        market_data = pd.DataFrame({
            'timestamp': dates[:len(prices)],
            'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(1000, 10000) * p for p in prices]
        })
        
        logger.info(f"✅ Generated {len(market_data)} days of market data")
        logger.info("   Regime distribution:")
        for start_day, regime in regime_changes:
            end_day = regime_changes[regime_changes.index((start_day, regime)) + 1][0] if regime_changes.index((start_day, regime)) < len(regime_changes) - 1 else n_points
            duration = end_day - start_day
            logger.info(f"     {regime}: Days {start_day}-{end_day} ({duration} days)")
        
        # Test 1: Regime Detection
        logger.info(f"\n{'='*50}")
        logger.info("🎭 Test 1: Market Regime Detection")
        logger.info(f"{'='*50}")
        
        # Detect regimes on recent data
        recent_data = market_data.tail(100)  # Last 100 days
        regime_result = regime_detector.detect_current_regime(recent_data[['close']])
        
        if regime_result:
            logger.info(f"   Detected regime: {regime_result.name}")
            logger.info(f"   Confidence: {regime_result.probability:.3f}")
            logger.info(f"   Characteristics: {regime_result.characteristics}")
        
        # Test 2: Edge Feature Engineering
        logger.info(f"\n{'='*50}")
        logger.info("⚙️ Test 2: Edge Feature Engineering")
        logger.info(f"{'='*50}")
        
        # Prepare edge features
        regime_info = {
            'name': regime_result.name if regime_result else 'STABLE',
            'probability': regime_result.probability if regime_result else 0.5
        } if regime_result else None
        
        edge_features = edge_clf.prepare_edge_features(
            market_data=market_data,
            regime_info=regime_info,
            forecast_data=None  # Could integrate with forecast service
        )
        
        logger.info(f"   Generated {len(edge_features.columns)} edge features")
        logger.info(f"   Feature samples: {len(edge_features)}")
        logger.info("   Sample features:")
        for feature in edge_features.columns[:5]:  # Show first 5 features
            values = edge_features[feature].tail(3)  # Last 3 values
            logger.info(f"     {feature}: {[f'{v:.3f}' for v in values]}")
        
        # Test 3: Meta-Label Creation
        logger.info(f"\n{'='*50}")
        logger.info("🏷️ Test 3: Meta-Label Creation")
        logger.info(f"{'='*50}")
        
        # Create meta-labels for training
        meta_labels = edge_clf.create_meta_labels(market_data, holding_period=5)
        logger.info(f"   Created {len(meta_labels)} meta-labels")
        logger.info(f"   Positive edges: {meta_labels.sum()} ({meta_labels.mean()*100:.1f}%)")
        logger.info(f"   Label distribution: {meta_labels.value_counts().to_dict()}")
        
        # Test 4: Model Training
        logger.info(f"\n{'='*50}")
        logger.info("🧠 Test 4: Edge Classifier Training")
        logger.info(f"{'='*50}")
        
        # Train the edge classifier
        logger.info(f"   Features shape: {edge_features.shape}")
        logger.info(f"   Labels shape: {meta_labels.shape}")
        logger.info(f"   Features index: {edge_features.index[:3]}...")
        logger.info(f"   Labels index: {meta_labels.index[:3]}...")
        logger.info(f"   Common indices: {len(edge_features.index.intersection(meta_labels.index))}")
        
        if len(edge_features) > len(meta_labels):
            # Align features with labels (remove last few feature rows)
            aligned_features = edge_features.iloc[:len(meta_labels)]
        else:
            aligned_features = edge_features
            meta_labels = meta_labels.iloc[:len(aligned_features)]
        
        logger.info(f"   Final training data shape: {aligned_features.shape}")
        logger.info(f"   Label balance: {meta_labels.mean()*100:.1f}% positive")
        
        # Train model
        cv_results = edge_clf.train(aligned_features, meta_labels)
        
        logger.info("   Cross-validation results:")
        for model_name, results in cv_results.items():
            logger.info(f"     {model_name}: AUC {results['mean_cv_score']:.3f} ± {results['std_cv_score']:.3f}")
        
        # Show feature importance
        if edge_clf.feature_importance:
            logger.info("   Top 5 most important features:")
            sorted_features = sorted(edge_clf.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                logger.info(f"     {feature}: {importance:.3f}")
        
        # Test 5: Edge Prediction
        logger.info(f"\n{'='*50}")
        logger.info("🔮 Test 5: Edge Prediction")
        logger.info(f"{'='*50}")
        
        # Make predictions on recent data
        recent_features = aligned_features.tail(20)  # Last 20 samples
        predictions = edge_clf.predict_edge(recent_features)
        
        logger.info(f"   Made predictions for {len(predictions)} samples")
        logger.info("   Recent edge probabilities:")
        
        for idx, (_, row) in enumerate(predictions.tail(5).iterrows()):
            logger.info(f"     Sample {idx+1}: {row['ensemble_prob']:.3f} "
                       f"(Pred: {'EDGE' if row['ensemble_pred'] == 1 else 'NO EDGE'})")
        
        # Test 6: Model Evaluation
        logger.info(f"\n{'='*50}")
        logger.info("📊 Test 6: Model Performance Evaluation")
        logger.info(f"{'='*50}")
        
        # Evaluate on recent data
        recent_labels = meta_labels.tail(len(predictions))
        evaluation = edge_clf.evaluate_model(recent_features, recent_labels)
        
        if evaluation:
            logger.info(f"   AUC-ROC Score: {evaluation['auc_roc']:.3f}")
            
            # Classification metrics
            class_report = evaluation['classification_report']
            if '1' in class_report:  # If there are positive samples
                logger.info(f"   Precision (edges): {class_report['1']['precision']:.3f}")
                logger.info(f"   Recall (edges): {class_report['1']['recall']:.3f}")
                logger.info(f"   F1-Score (edges): {class_report['1']['f1-score']:.3f}")
            
            logger.info(f"   Accuracy: {class_report['accuracy']:.3f}")
        
        # Test 7: Integration with Risk Engine
        logger.info(f"\n{'='*50}")
        logger.info("🛡️ Test 7: Risk-Adjusted Edge Processing")
        logger.info(f"{'='*50}")
        
        # Process edges through risk engine
        for idx, (_, pred_row) in enumerate(predictions.tail(3).iterrows()):
            edge_probability = pred_row['ensemble_prob']
            has_edge = pred_row['ensemble_pred'] == 1
            
            if has_edge:
                logger.info(f"   Edge {idx+1} detected (Probability: {edge_probability:.3f})")
                
                # Simulate position assessment
                simulated_position = {
                    'symbol': 'BTC/USDT',
                    'entry_price': market_data['close'].iloc[-1],
                    'position_size': 0.1,  # 0.1 BTC
                    'stop_loss': market_data['close'].iloc[-1] * 0.95,  # 5% SL
                    'take_profit': market_data['close'].iloc[-1] * 1.10,  # 10% TP
                    'volatility': 0.03,
                    'regime': regime_result.name if regime_result else 'STABLE'
                }
                
                risk_assessment = risk_engine.assess_position_risk(**simulated_position)
                
                if risk_assessment['approved']:
                    logger.info(f"     ✅ Risk-approved edge trade")
                    logger.info(f"     Recommended size: {risk_assessment['risk_metrics']['position_percentage']*100:.2f}% of capital")
                else:
                    logger.info(f"     ❌ Risk-rejected despite edge detection")
                    logger.info(f"     Reason: {edge_clf._get_rejection_summary(risk_assessment)}")
            else:
                logger.info(f"   No edge detected (Probability: {edge_probability:.3f})")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("🎯 EDGE CLASSIFICATION DEMO COMPLETED")
        logger.info(f"{'='*60}")
        logger.info("✅ Key achievements:")
        logger.info("   • Implemented edge-focused classification (not price prediction)")
        logger.info("   • Created regime-aware feature engineering")
        logger.info("   • Built meta-labeling system for genuine edge detection")
        logger.info("   • Trained ensemble classifier with cross-validation")
        logger.info("   • Integrated with risk engine for approval workflow")
        logger.info("   • Demonstrated full edge-to-trade pipeline")
        
        logger.info(f"\n🚀 Chloe 0.4 Progress:")
        logger.info("   ✅ Phase 1: Market Intelligence Layer (70% complete)")
        logger.info("   ✅ Phase 2: Risk Engine Core Enhancement (complete)")
        logger.info("   ✅ Phase 3: Edge Classification Model (now complete)")
        logger.info("   ⬜ Phase 4: Portfolio Construction Logic")
        logger.info("   ⬜ Phase 5: Simulation Lab")
        
        # Performance summary
        if evaluation and 'auc_roc' in evaluation:
            logger.info(f"\n📈 Edge Classifier Performance:")
            logger.info(f"   AUC-ROC: {evaluation['auc_roc']:.3f}")
            logger.info(f"   Accuracy: {evaluation['classification_report']['accuracy']:.3f}")
            if '1' in evaluation['classification_report']:
                logger.info(f"   Edge Detection Rate: {evaluation['classification_report']['1']['recall']*100:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def _get_rejection_summary(self, risk_assessment: dict) -> str:
    """Helper method to summarize risk rejection reasons"""
    criteria = risk_assessment.get('approval_criteria', {})
    failed_criteria = [k for k, v in criteria.items() if not v]
    
    if not failed_criteria:
        return "Unknown reason"
    
    reason_map = {
        'size_limit': 'position too large',
        'drawdown_limit': 'excessive drawdown risk',
        'rr_ratio': 'poor risk/reward ratio',
        'kelly_compliance': 'deviates from optimal sizing',
        'volatility_check': 'high volatility'
    }
    
    reasons = [reason_map.get(crit, crit) for crit in failed_criteria]
    return ', '.join(reasons)

# Add helper method to edge classifier class
import types
EdgeClassifier._get_rejection_summary = types.MethodType(_get_rejection_summary, EdgeClassifier)

if __name__ == "__main__":
    asyncio.run(demo_edge_classification())