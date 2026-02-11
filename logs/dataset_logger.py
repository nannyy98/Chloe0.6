"""
Learning Dataset Logger for Adaptive Institutional AI Trader
Records all trading decisions for future model improvement
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class DecisionDatasetLogger:
    """Logger for recording trading decisions and outcomes"""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = Path(log_directory)
        self.dataset_path = self.log_directory / "decision_dataset.parquet"
        self.decision_log_path = self.log_directory / "decision_log.jsonl"
        
        # Ensure directories exist
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset
        self.dataset = pd.DataFrame()
        self.buffer = []  # Buffer for efficient logging
        self.buffer_size = 100  # Log every N records
        
        # Columns for the dataset
        self.feature_columns = [
            # Market microstructure features
            'feature_return_1d', 'feature_return_5d', 'feature_return_20d',
            'feature_rv_5d', 'feature_rv_20d', 'feature_rv_60d',
            'feature_vol_regime', 'feature_vol_clustering',
            'feature_volume_ratio', 'feature_vol_price_corr',
            
            # Technical features
            'feature_rsi_14', 'feature_price_position_20',
            'feature_momentum_10', 'feature_stoch_14',
            
            # Regime features
            'feature_trend_strength_20', 'feature_regime_state',
            'feature_price_efficiency', 'feature_spread_proxy'
        ]
        
        # Decision-related columns
        self.decision_columns = [
            'symbol', 'decision_type', 'forecast_p10', 'forecast_p50', 'forecast_p90',
            'forecast_expected_return', 'forecast_volatility', 'forecast_confidence',
            'regime_detected', 'regime_confidence', 'position_size',
            'entry_price', 'stop_loss', 'take_profit', 'strategy_id',
            'allocation_weight', 'risk_adjustment', 'confidence_adjustment'
        ]
        
        # Outcome columns
        self.outcome_columns = [
            'exit_price', 'realized_pnl', 'realized_return', 'holding_period',
            'execution_quality', 'slippage', 'commission', 'outcome_regime',
            'model_accuracy', 'calibration_error'
        ]
        
        logger.info(f"📊 Decision Dataset Logger initialized: {self.dataset_path}")
    
    def log_decision(self, 
                    features: Dict[str, float],
                    forecast: Dict[str, float],
                    regime_info: Dict[str, Any],
                    decision: Dict[str, Any],
                    timestamp: datetime = None) -> bool:
        """
        Log a trading decision with all relevant information
        
        Args:
            features: Market microstructure and technical features
            forecast: Forecast from quantile model
            regime_info: Regime detection results
            decision: Trading decision details
            timestamp: When decision was made
            
        Returns:
            True if logged successfully
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Create record
            record = {
                'timestamp': timestamp,
                'features': features,
                'forecast': forecast,
                'regime_info': regime_info,
                'decision': decision,
                'logged_at': datetime.now()
            }
            
            # Add to buffer
            self.buffer.append(record)
            
            # Write to file if buffer is full
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
            
            logger.debug(f"📝 Decision logged for {decision.get('symbol', 'UNKNOWN')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to log decision: {e}")
            return False
    
    def _flush_buffer(self):
        """Write buffered records to persistent storage"""
        try:
            if not self.buffer:
                return
            
            # Process records
            records_data = []
            for record in self.buffer:
                flat_record = self._flatten_record(record)
                records_data.append(flat_record)
            
            # Convert to DataFrame
            new_data = pd.DataFrame(records_data)
            
            # Save to JSONL (for streaming)
            with open(self.decision_log_path, 'a') as f:
                for record in records_data:
                    f.write(json.dumps(record) + '\n')
            
            # Save to Parquet (for analysis)
            if self.dataset.empty:
                self.dataset = new_data
            else:
                self.dataset = pd.concat([self.dataset, new_data], ignore_index=True)
            
            # Save to parquet
            self.dataset.to_parquet(self.dataset_path)
            
            logger.info(f"💾 Flushed {len(self.buffer)} records to dataset")
            
            # Clear buffer
            self.buffer = []
            
        except Exception as e:
            logger.error(f"❌ Failed to flush buffer: {e}")
    
    def _flatten_record(self, record: Dict) -> Dict[str, Any]:
        """Flatten nested record structure for DataFrame"""
        flat_record = {}
        
        # Add timestamp
        flat_record['timestamp'] = record['timestamp']
        flat_record['logged_at'] = record['logged_at']
        
        # Flatten features
        features = record.get('features', {})
        for key, value in features.items():
            flat_record[f'feature_{key}'] = value
        
        # Flatten forecast
        forecast = record.get('forecast', {})
        for key, value in forecast.items():
            flat_record[f'forecast_{key}'] = value
        
        # Flatten regime info
        regime_info = record.get('regime_info', {})
        for key, value in regime_info.items():
            flat_record[f'regime_{key}'] = value
        
        # Flatten decision
        decision = record.get('decision', {})
        for key, value in decision.items():
            flat_record[f'decision_{key}'] = value
        
        return flat_record
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the logged dataset"""
        try:
            if self.dataset.empty and self.dataset_path.exists():
                # Load dataset if not in memory
                self.dataset = pd.read_parquet(self.dataset_path)
            
            if self.dataset.empty:
                return {
                    'total_records': 0,
                    'date_range': None,
                    'symbols': [],
                    'strategies': []
                }
            
            summary = {
                'total_records': len(self.dataset),
                'date_range': {
                    'start': self.dataset['timestamp'].min().isoformat(),
                    'end': self.dataset['timestamp'].max().isoformat()
                },
                'symbols': self.dataset.filter(regex='decision_symbol').dropna().unique().tolist(),
                'strategies': self.dataset.filter(regex='decision_strategy').dropna().unique().tolist(),
                'feature_columns': [col for col in self.dataset.columns if col.startswith('feature_')],
                'forecast_columns': [col for col in self.dataset.columns if col.startswith('forecast_')],
                'regime_columns': [col for col in self.dataset.columns if col.startswith('regime_')],
                'decision_columns': [col for col in self.dataset.columns if col.startswith('decision_')]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get dataset summary: {e}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from logged decisions"""
        try:
            if self.dataset.empty and self.dataset_path.exists():
                self.dataset = pd.read_parquet(self.dataset_path)
            
            if self.dataset.empty:
                return {}
            
            # Look for outcome columns
            outcome_cols = [col for col in self.dataset.columns if col.startswith('decision_realized_')]
            
            if not outcome_cols:
                return {'message': 'No outcome data available yet'}
            
            # Calculate metrics
            realized_returns = self.dataset.filter(regex='decision_realized_return').dropna()
            if len(realized_returns) == 0:
                return {'message': 'No realized returns available'}
            
            # Get the first return column (assuming single return column)
            return_col = outcome_cols[0]
            returns = self.dataset[return_col].dropna()
            
            metrics = {
                'total_trades': len(returns),
                'avg_return': float(returns.mean()),
                'std_return': float(returns.std()),
                'sharpe_ratio': float(returns.mean() / returns.std()) if returns.std() != 0 else 0.0,
                'win_rate': float((returns > 0).mean()),
                'max_win': float(returns.max()) if len(returns) > 0 else 0.0,
                'max_loss': float(returns.min()) if len(returns) > 0 else 0.0,
                'total_pnl': float(returns.sum()),
                'profit_factor': self._calculate_profit_factor(returns)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        gross_profit = positive_returns.sum() if len(positive_returns) > 0 else 0
        gross_loss = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_feature_importance_from_decisions(self) -> Dict[str, float]:
        """Analyze which features are most predictive of profitable decisions"""
        try:
            if self.dataset.empty and self.dataset_path.exists():
                self.dataset = pd.read_parquet(self.dataset_path)
            
            if self.dataset.empty:
                return {}
            
            # Get feature columns and outcome columns
            feature_cols = [col for col in self.dataset.columns if col.startswith('feature_')]
            outcome_cols = [col for col in self.dataset.columns if col.startswith('decision_realized_')]
            
            if not feature_cols or not outcome_cols:
                return {}
            
            # For now, return basic correlation analysis
            outcome_col = outcome_cols[0]  # Use first outcome column
            returns = self.dataset[outcome_col].dropna()
            
            correlations = {}
            for col in feature_cols:
                feature_data = self.dataset[col].dropna()
                # Align indices
                aligned_data = pd.concat([returns, feature_data], axis=1).dropna()
                
                if len(aligned_data) > 10:  # Minimum samples for correlation
                    corr = aligned_data[outcome_col].corr(aligned_data[col])
                    if not pd.isna(corr):
                        correlations[col] = abs(float(corr))  # Use absolute correlation
            
            # Sort by importance
            sorted_correlations = dict(
                sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:20]  # Top 20
            )
            
            return sorted_correlations
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate feature importance: {e}")
            return {}
    
    def export_for_model_training(self, 
                                target_column: str = 'decision_realized_return',
                                feature_prefix: str = 'feature_') -> Tuple[pd.DataFrame, pd.Series]:
        """Export cleaned dataset for model training"""
        try:
            if self.dataset.empty and self.dataset_path.exists():
                self.dataset = pd.read_parquet(self.dataset_path)
            
            if self.dataset.empty:
                return pd.DataFrame(), pd.Series(dtype=float)
            
            # Select feature columns
            feature_cols = [col for col in self.dataset.columns if col.startswith(feature_prefix)]
            
            # Select target column
            target_cols = [col for col in self.dataset.columns if target_column in col]
            if not target_cols:
                logger.warning(f"Target column {target_column} not found")
                return self.dataset[feature_cols], pd.Series(dtype=float)
            
            target_col = target_cols[0]
            
            # Combine and clean data
            data = pd.concat([
                self.dataset[feature_cols],
                self.dataset[[target_col]]
            ], axis=1).dropna()
            
            if data.empty:
                return pd.DataFrame(), pd.Series(dtype=float)
            
            X = data[feature_cols]
            y = data[target_col]
            
            logger.info(f"📈 Exported {len(X)} samples for training with {len(feature_cols)} features")
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Failed to export for training: {e}")
            return pd.DataFrame(), pd.Series(dtype=float)
    
    def cleanup_old_logs(self, days_to_keep: int = 365):
        """Remove logs older than specified days"""
        try:
            if self.dataset.empty and self.dataset_path.exists():
                self.dataset = pd.read_parquet(self.dataset_path)
            
            if not self.dataset.empty:
                cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
                self.dataset = self.dataset[self.dataset['timestamp'] >= cutoff_date]
                self.dataset.to_parquet(self.dataset_path)
                
                logger.info(f"🗑️ Cleaned up logs older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old logs: {e}")
    
    def force_flush(self):
        """Force write all buffered records to disk"""
        if self.buffer:
            self._flush_buffer()
        logger.info("💾 Forced flush of all buffered records")
    
    def get_decision_frequency(self) -> Dict[str, int]:
        """Get frequency of different decision types"""
        try:
            if self.dataset.empty and self.dataset_path.exists():
                self.dataset = pd.read_parquet(self.dataset_path)
            
            if self.dataset.empty:
                return {}
            
            # Get decision type column
            decision_type_cols = [col for col in self.dataset.columns if 'decision_type' in col]
            if not decision_type_cols:
                return {}
            
            decision_type_col = decision_type_cols[0]
            value_counts = self.dataset[decision_type_col].value_counts()
            
            return dict(value_counts)
            
        except Exception as e:
            logger.error(f"❌ Failed to get decision frequency: {e}")
            return {}

class SelfLearningMonitor:
    """Monitor for self-improvement and learning from past decisions"""
    
    def __init__(self, dataset_logger: DecisionDatasetLogger):
        self.logger = dataset_logger
        self.performance_threshold = 0.01  # Minimum acceptable performance
        self.learning_interval = 30  # Days between model updates
        self.last_learning_date = datetime.now()
        
        logger.info("🤖 Self-Learning Monitor initialized")
    
    async def evaluate_model_performance(self) -> Dict[str, Any]:
        """Evaluate if current models are performing adequately"""
        try:
            # Get performance metrics
            metrics = self.logger.get_performance_metrics()
            
            if 'error' in metrics:
                return {'status': 'error', 'message': metrics['error']}
            
            if not metrics or 'total_trades' not in metrics:
                return {'status': 'insufficient_data', 'message': 'Not enough data to evaluate'}
            
            # Evaluate performance
            sharpe = metrics.get('sharpe_ratio', 0)
            win_rate = metrics.get('win_rate', 0)
            total_trades = metrics.get('total_trades', 0)
            
            evaluation = {
                'status': 'needs_attention' if sharpe < 0.5 or win_rate < 0.5 else 'performing_well',
                'sharpe_ratio': sharpe,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'should_update_model': sharpe < 0.3 or total_trades > 100,  # Update if poor performance or enough data
                'recommendation': self._get_recommendation(sharpe, win_rate, total_trades)
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate model performance: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _get_recommendation(self, sharpe: float, win_rate: float, total_trades: int) -> str:
        """Get recommendation based on performance"""
        if total_trades < 10:
            return "Collect more data before evaluation"
        elif sharpe < 0:
            return "Model is consistently losing money - consider disabling"
        elif sharpe < 0.3:
            return "Poor performance - model update recommended"
        elif sharpe < 0.7:
            return "Average performance - monitor closely"
        else:
            return "Good performance - continue current approach"
    
    async def generate_insights(self) -> Dict[str, Any]:
        """Generate insights from logged decisions"""
        try:
            summary = self.logger.get_dataset_summary()
            metrics = self.logger.get_performance_metrics()
            feature_importance = self.logger.get_feature_importance_from_decisions()
            
            insights = {
                'dataset_summary': summary,
                'performance_metrics': metrics,
                'key_features': list(feature_importance.keys())[:10],  # Top 10 features
                'feature_correlations': dict(list(feature_importance.items())[:10]),
                'recommendations': await self._generate_recommendations(metrics, feature_importance)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Failed to generate insights: {e}")
            return {'error': str(e)}
    
    async def _generate_recommendations(self, metrics: Dict, feature_importance: Dict) -> List[str]:
        """Generate recommendations based on performance and features"""
        recommendations = []
        
        if 'sharpe_ratio' in metrics:
            sharpe = metrics['sharpe_ratio']
            if sharpe < 0.3:
                recommendations.append("Consider reducing position sizes to minimize losses")
            elif sharpe > 1.0:
                recommendations.append("Strong performance - consider increasing position sizes cautiously")
        
        if 'win_rate' in metrics:
            win_rate = metrics['win_rate']
            if win_rate < 0.4:
                recommendations.append("Low win rate - investigate entry criteria")
            elif win_rate > 0.7:
                recommendations.append("High win rate - consider if returns are maximized")
        
        if feature_importance:
            top_features = list(feature_importance.keys())[:3]
            recommendations.append(f"Key predictive features: {', '.join(top_features[:2])}")
        
        if not recommendations:
            recommendations.append("Performance looks stable - continue current approach")
        
        return recommendations