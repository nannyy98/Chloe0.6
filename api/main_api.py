"""
Main API for Chloe AI
Provides endpoints for market analysis, signals, and portfolio management
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import logging
from datetime import datetime
import os

# Import our modules
from data.data_agent import DataAgent
from indicators.indicator_calculator import IndicatorCalculator
from models.ml_core import MLSignalsCore, SignalProcessor
from risk.risk_engine import RiskEngine, TradeSignal, RiskLevel
from llm.chloe_llm import ChloeLLM, SignalAnalysis
from backtest.backtester import Backtester
from realtime.api_endpoints import router as realtime_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chloe AI API",
    description="AI-powered market analysis and trading signals",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(realtime_router)

# Initialize core components
data_agent = DataAgent()
indicator_calc = IndicatorCalculator()
ml_core = MLSignalsCore()
signal_processor = SignalProcessor()
risk_engine = RiskEngine()
chloe_llm = ChloeLLM()
backtester = Backtester()

# Request/Response models
class SymbolRequest(BaseModel):
    symbol: str
    exchange: str = "binance"
    days: int = 365

class SignalRequest(BaseModel):
    symbol: str
    model_type: str = "xgboost"
    look_ahead: int = 5

class BacktestRequest(BaseModel):
    symbol: str
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001

class RiskAssessmentRequest(BaseModel):
    symbol: str
    signal: str
    entry_price: float
    stop_loss: float
    take_profit: float
    volatility: float

class PortfolioAllocationRequest(BaseModel):
    symbols: List[str]
    total_capital: float
    max_allocation_per_asset: float = 0.1

class AnalysisResponse(BaseModel):
    symbol: str
    current_price: float
    indicators: Dict
    signal: str
    confidence: float
    risk_level: str
    explanation: str
    timestamp: str

class SignalResponse(BaseModel):
    symbol: str
    signal: str
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    risk_level: str
    explanation: str
    suggested_action: str

class BacktestResponse(BaseModel):
    symbol: str
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Chloe AI API",
        "version": "0.1.0",
        "endpoints": [
            "/analyze/{symbol}",
            "/signals/{symbol}",
            "/backtest/{symbol}",
            "/risk-assess",
            "/portfolio-optimize"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "data_agent": "ready",
            "indicator_calc": "ready", 
            "ml_core": "ready",
            "risk_engine": "ready",
            "chloe_llm": "ready"
        }
    }

@app.get("/analyze/{symbol}")
async def analyze_symbol(symbol: str, days: int = 365):
    """
    Analyze a symbol and return comprehensive market analysis
    """
    try:
        logger.info(f"Analyzing symbol: {symbol}")
        
        # Fetch data
        data = await data_agent.fetch_crypto_ohlcv(symbol, timeframe='1d', limit=days)
        if data is None:
            # Try as stock
            data = await data_agent.fetch_stock_ohlcv(symbol, period=f'{days}d', interval='1d')
            if data is None:
                raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
        
        # Calculate indicators
        data_with_indicators = indicator_calc.calculate_all_indicators(data)
        
        # Get latest price
        current_price = data['close'].iloc[-1] if 'close' in data.columns else data['Close'].iloc[-1]
        
        # Prepare features for ML model
        X, y = ml_core.prepare_features_and_target(data_with_indicators, lookahead_period=5)
        
        # Train model temporarily (in production, you'd use a pre-trained model)
        if len(X) > 10:  # Need enough data to train
            ml_core.train(X, y)
            
            # Predict signals
            latest_features = X.tail(1)
            predictions, probabilities = ml_core.predict_with_probabilities(latest_features)
            
            # Process predictions
            signals_df = signal_processor.process_predictions(predictions, probabilities)
            signal = signals_df['signal'].iloc[0]
            confidence = signals_df['confidence'].iloc[0]
            
            # Calculate risk metrics
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1] if 'close' in data.columns else data['Close'].pct_change().rolling(20).std().iloc[-1]
            
            # Assess risk level
            if confidence > 0.8 and volatility < 0.03:
                risk_level = "LOW"
            elif confidence > 0.6 or volatility > 0.05:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            # Generate explanation using LLM
            technical_data = {col: data_with_indicators[col].iloc[-1] for col in data_with_indicators.columns if pd.api.types.is_numeric_dtype(data_with_indicators[col])}
            risk_data = {'volatility': volatility}
            
            analysis = chloe_llm.analyze_signal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                technical_data=technical_data,
                risk_data=risk_data
            )
            
            response = AnalysisResponse(
                symbol=symbol,
                current_price=current_price,
                indicators={},
                signal=signal,
                confidence=confidence,
                risk_level=risk_level,
                explanation=analysis.explanation,
                timestamp=datetime.now().isoformat()
            )
            
            # Add key indicators to response
            for indicator in ['rsi_14', 'macd', 'ema_20', 'ema_50', 'volatility']:
                if indicator in data_with_indicators.columns:
                    val = data_with_indicators[indicator].iloc[-1]
                    if pd.notna(val):
                        response.indicators[indicator] = round(float(val), 4)
            
            return response
        else:
            # Not enough data to train model, return just indicators
            latest_indicators = {}
            for indicator in ['rsi_14', 'macd', 'ema_20', 'ema_50', 'volatility']:
                if indicator in data_with_indicators.columns:
                    val = data_with_indicators[indicator].iloc[-1]
                    if pd.notna(val):
                        latest_indicators[indicator] = round(float(val), 4)
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "indicators": latest_indicators,
                "signal": "INSUFFICIENT_DATA",
                "confidence": 0.0,
                "risk_level": "UNKNOWN",
                "explanation": "Not enough data to generate reliable signals. More historical data is needed for accurate analysis.",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error analyzing symbol {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/signals")
async def generate_signals(request: SignalRequest):
    """
    Generate trading signals for a symbol
    """
    try:
        logger.info(f"Generating signals for: {request.symbol}")
        
        # Fetch data
        data = await data_agent.fetch_crypto_ohlcv(request.symbol, timeframe='1d', limit=365)
        if data is None:
            data = await data_agent.fetch_stock_ohlcv(request.symbol, period='1y', interval='1d')
            if data is None:
                raise HTTPException(status_code=404, detail=f"No data found for symbol: {request.symbol}")
        
        # Calculate indicators
        data_with_indicators = indicator_calc.calculate_all_indicators(data)
        
        # Prepare features for ML model
        X, y = ml_core.prepare_features_and_target(data_with_indicators, lookahead_period=request.look_ahead)
        
        # Train model temporarily (in production, you'd use a pre-trained model)
        if len(X) > 10:
            ml_core.train(X, y)
            
            # Predict signals
            latest_features = X.tail(1)
            predictions, probabilities = ml_core.predict_with_probabilities(latest_features)
            
            # Process predictions
            signals_df = signal_processor.process_predictions(predictions, probabilities)
            signal = signals_df['signal'].iloc[0]
            confidence = signals_df['confidence'].iloc[0]
            
            # Calculate risk metrics
            current_price = data['close'].iloc[-1] if 'close' in data.columns else data['Close'].iloc[-1]
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1] if 'close' in data.columns else data['Close'].pct_change().rolling(20).std().iloc[-1]
            
            # Calculate stop loss and take profit
            atr = data['high'].rolling(14).max().iloc[-1] - data['low'].rolling(14).min().iloc[-1] if 'high' in data.columns and 'low' in data.columns else volatility * current_price * 3
            stop_loss, take_profit = risk_engine.calculate_stop_loss_take_profit(current_price, signal, atr)
            
            # Calculate position size
            position_size = risk_engine.calculate_position_size(current_price, stop_loss, 10000, 0.02)  # Assuming $10k account
            
            # Assess risk level
            if confidence > 0.8 and volatility < 0.03:
                risk_level = "LOW"
            elif confidence > 0.6 or volatility > 0.05:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            # Generate explanation using LLM
            technical_data = {col: data_with_indicators[col].iloc[-1] for col in data_with_indicators.columns if pd.api.types.is_numeric_dtype(data_with_indicators[col])}
            risk_data = {'volatility': volatility}
            
            analysis = chloe_llm.analyze_signal(
                symbol=request.symbol,
                signal=signal,
                confidence=confidence,
                technical_data=technical_data,
                risk_data=risk_data
            )
            
            response = SignalResponse(
                symbol=request.symbol,
                signal=signal,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_level=risk_level,
                explanation=analysis.explanation,
                suggested_action=analysis.suggested_action
            )
            
            return response
        else:
            raise HTTPException(status_code=400, detail="Not enough data to generate signals")
            
    except Exception as e:
        logger.error(f"Error generating signals for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest/{symbol}")
async def backtest_strategy(request: BacktestRequest):
    """
    Backtest trading strategy on historical data
    """
    try:
        logger.info(f"Backtesting strategy for: {request.symbol}")
        
        # Fetch data
        data = await data_agent.fetch_crypto_ohlcv(request.symbol, timeframe='1d', limit=365)
        if data is None:
            data = await data_agent.fetch_stock_ohlcv(request.symbol, period='1y', interval='1d')
            if data is None:
                raise HTTPException(status_code=404, detail=f"No data found for symbol: {request.symbol}")
        
        # Calculate indicators
        data_with_indicators = indicator_calc.calculate_all_indicators(data)
        
        # Generate simple signals based on RSI for demonstration
        # In a real scenario, you'd use your ML model
        signals = pd.Series(0, index=data.index)  # Default to hold
        
        # Simple RSI-based signals
        if 'rsi_14' in data_with_indicators.columns:
            rsi = data_with_indicators['rsi_14']
            signals[rsi < 30] = 1  # Buy when RSI < 30 (oversold)
            signals[rsi > 70] = -1  # Sell when RSI > 70 (overbought)
        
        # Run backtest
        backtester = Backtester(initial_capital=request.initial_capital)
        results = backtester.run_backtest(data, signals, transaction_cost=request.transaction_cost)
        
        response = BacktestResponse(
            symbol=request.symbol,
            initial_capital=results['initial_capital'],
            final_capital=results['final_capital'],
            total_return=results['total_return'],
            annualized_return=results['annualized_return'],
            sharpe_ratio=results['sharpe_ratio'],
            max_drawdown=results['max_drawdown'],
            win_rate=results['win_rate'],
            num_trades=results['num_trades']
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error backtesting strategy for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk-assess")
async def assess_risk(request: RiskAssessmentRequest):
    """
    Assess risk of a potential trade
    """
    try:
        logger.info(f"Assessing risk for: {request.symbol}")
        
        # Create trade signal object
        signal = TradeSignal(
            symbol=request.symbol,
            signal=request.signal,
            confidence=0.7,  # Default confidence
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            position_size=0.0,
            risk_level=RiskLevel.MEDIUM,  # Will be updated
            volatility=request.volatility,
            timestamp=pd.Timestamp.now()
        )
        
        # Assess risk using risk engine
        risk_level, risk_metrics = risk_engine.assess_signal_risk(signal)
        
        # Validate trade
        is_valid, reason, validation_details = risk_engine.validate_trade(signal)
        
        response = {
            "symbol": request.symbol,
            "risk_level": risk_level.value,
            "risk_metrics": risk_metrics,
            "trade_valid": is_valid,
            "validation_reason": reason,
            "validation_details": validation_details,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error assessing risk for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols/{exchange}")
async def get_symbols(exchange: str = "binance"):
    """
    Get available symbols from exchange
    """
    try:
        if exchange == "binance":
            # In a real implementation, this would fetch from the exchange
            symbols = [
                "BTC/USDT", "ETH/USDT", "ADA/USDT", "BNB/USDT", "SOL/USDT",
                "XRP/USDT", "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "SHIB/USDT"
            ]
        else:
            # Default to some common symbols
            symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        
        return {"exchange": exchange, "symbols": symbols}
    except Exception as e:
        logger.error(f"Error getting symbols for {exchange}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event to initialize components
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Chloe AI API...")
    logger.info("✅ API is ready to serve requests")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down Chloe AI API...")

# Example usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)