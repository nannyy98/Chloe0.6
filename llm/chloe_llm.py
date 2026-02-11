"""
Chloe LLM Module
Handles natural language processing and explanation of trading signals
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SignalAnalysis:
    """Represents an analyzed trading signal with explanation"""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    explanation: str
    risk_assessment: str
    market_conditions: str
    suggested_action: str
    timestamp: str

class ChloeLLM:
    """
    Chloe's Language Model interface for explaining trading signals
    """
    
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
        self.model_provider = model_provider
        self.model_name = model_name
        self.is_initialized = False
        
        # Mock model for demonstration purposes
        self.mock_responses = {
            "BUY": {
                "explanation": "Technical indicators show positive momentum with RSI indicating moderate bullish sentiment. "
                              "Price has broken above key resistance levels with increasing volume.",
                "risk_assessment": "Medium risk with stop loss recommended at support level. "
                                 "Volatility is elevated but within normal ranges.",
                "market_conditions": "Bullish trend with strong volume confirmation. "
                                   "MACD showing positive divergence.",
                "suggested_action": "Consider entering position with 2% of capital allocation. "
                                  "Set stop loss at recent support level."
            },
            "SELL": {
                "explanation": "Bearish divergence detected in momentum indicators. "
                              "Price approaching resistance with decreasing volume.",
                "risk_assessment": "Medium-high risk as market shows signs of reversal. "
                                 "Protective measures recommended.",
                "market_conditions": "Potential trend reversal with weakening momentum. "
                                   "Volume declining on recent rallies.",
                "suggested_action": "Consider reducing position size or exiting if holding. "
                                  "Set stop loss above recent resistance."
            },
            "HOLD": {
                "explanation": "Mixed signals from technical indicators. "
                              "Market in consolidation phase with no clear directional bias.",
                "risk_assessment": "Low risk but low opportunity. "
                                 "Market uncertainty is high.",
                "market_conditions": "Sideways market with no clear trend. "
                                   "Waiting for clearer directional cues.",
                "suggested_action": "Maintain current position or remain in cash. "
                                  "Wait for stronger directional signals."
            }
        }
        
        logger.info("🤖 Chloe LLM module initialized")
        self.is_initialized = True
    
    def analyze_signal(self, symbol: str, signal: str, confidence: float, 
                      technical_data: Dict, risk_data: Dict) -> SignalAnalysis:
        """
        Analyze a trading signal and provide human-readable explanation
        
        Args:
            symbol: Trading symbol
            signal: Trading signal (BUY/SELL/HOLD)
            confidence: Signal confidence (0.0 to 1.0)
            technical_data: Technical indicator values
            risk_data: Risk assessment data
            
        Returns:
            SignalAnalysis object with explanation
        """
        logger.info(f"🔍 Analyzing signal for {symbol}: {signal} with {confidence:.2f} confidence")
        
        # Get mock response based on signal type
        response_data = self.mock_responses.get(signal.upper(), self.mock_responses["HOLD"])
        
        # Create more detailed explanation based on technical data
        explanation = self._generate_detailed_explanation(
            signal, technical_data, risk_data, response_data["explanation"]
        )
        
        analysis = SignalAnalysis(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            explanation=explanation,
            risk_assessment=response_data["risk_assessment"],
            market_conditions=response_data["market_conditions"],
            suggested_action=response_data["suggested_action"],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Signal analysis completed for {symbol}")
        return analysis
    
    def _generate_detailed_explanation(self, signal: str, technical_data: Dict, 
                                     risk_data: Dict, base_explanation: str) -> str:
        """
        Generate a detailed explanation combining technical data and base explanation
        
        Args:
            signal: Trading signal
            technical_data: Technical indicator values
            risk_data: Risk assessment data
            base_explanation: Base explanation to enhance
            
        Returns:
            Enhanced explanation string
        """
        explanation_parts = [base_explanation]
        
        # Add specific technical readings
        if 'rsi_14' in technical_data:
            rsi_val = technical_data['rsi_14']
            if pd.notna(rsi_val):
                if signal == 'BUY' and rsi_val > 70:
                    explanation_parts.append(f"Note: RSI is at {rsi_val:.2f}, indicating potential overbought conditions.")
                elif signal == 'SELL' and rsi_val < 30:
                    explanation_parts.append(f"Note: RSI is at {rsi_val:.2f}, indicating potential oversold conditions.")
                else:
                    explanation_parts.append(f"RSI reading is {rsi_val:.2f}, supporting the {signal.lower()} signal.")
        
        # Add MACD information if available
        if 'macd' in technical_data and 'macd_signal' in technical_data:
            macd_val = technical_data['macd']
            macd_signal = technical_data['macd_signal']
            if pd.notna(macd_val) and pd.notna(macd_signal):
                if signal == 'BUY' and macd_val > macd_signal:
                    explanation_parts.append(f"MACD ({macd_val:.4f}) is above signal line ({macd_signal:.4f}), confirming bullish momentum.")
                elif signal == 'SELL' and macd_val < macd_signal:
                    explanation_parts.append(f"MACD ({macd_val:.4f}) is below signal line ({macd_signal:.4f}), confirming bearish momentum.")
        
        # Add volatility information
        if 'volatility' in risk_data:
            vol_val = risk_data['volatility']
            if pd.notna(vol_val):
                vol_desc = "high" if vol_val > 0.03 else "moderate" if vol_val > 0.015 else "low"
                explanation_parts.append(f"Current market volatility is {vol_desc} at {vol_val:.4f}.")
        
        return " ".join(explanation_parts)
    
    def explain_market_situation(self, symbol: str, market_data: Dict) -> str:
        """
        Provide general market analysis for a symbol
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            
        Returns:
            Market situation explanation
        """
        explanation = f"Current market analysis for {symbol}:\n\n"
        
        # Add price information
        if 'current_price' in market_data:
            explanation += f"- Current price: ${market_data['current_price']:.2f}\n"
        
        # Add trend information
        if 'trend' in market_data:
            explanation += f"- Market trend: {market_data['trend']}\n"
        
        # Add volatility information
        if 'volatility' in market_data:
            explanation += f"- Volatility level: {market_data['volatility']:.4f}\n"
        
        # Add volume information
        if 'volume' in market_data:
            explanation += f"- Trading volume: {market_data['volume']:.2f}\n"
        
        explanation += "\nThis analysis is for informational purposes only. "
        explanation += "Always consider your own risk tolerance and investment strategy."
        
        return explanation
    
    def generate_risk_warning(self, symbol: str, risk_level: str, risk_factors: List[str]) -> str:
        """
        Generate risk warning based on risk assessment
        
        Args:
            symbol: Trading symbol
            risk_level: Risk level (LOW/MEDIUM/HIGH/EXTREME)
            risk_factors: List of risk factors
            
        Returns:
            Risk warning message
        """
        warning_levels = {
            "LOW": "🟢 Low Risk: Normal market conditions. Standard precautions advised.",
            "MEDIUM": "🟡 Medium Risk: Some caution advised. Monitor positions closely.",
            "HIGH": "🟠 High Risk: Significant caution required. Consider reducing exposure.",
            "EXTREME": "🔴 Extreme Risk: Exercise extreme caution. Consider protective measures."
        }
        
        warning = f"Risk Assessment for {symbol}:\n\n"
        warning += f"{warning_levels.get(risk_level, warning_levels['MEDIUM'])}\n\n"
        
        if risk_factors:
            warning += "Risk Factors:\n"
            for factor in risk_factors:
                warning += f"- {factor}\n"
        
        warning += "\nRemember to follow your risk management strategy."
        
        return warning
    
    def answer_market_question(self, question: str, market_context: Dict) -> str:
        """
        Answer a natural language question about market conditions
        
        Args:
            question: Natural language question
            market_context: Contextual market data
            
        Returns:
            Answer to the question
        """
        # In a real implementation, this would connect to an LLM API
        # For now, we'll provide template responses based on common questions
        
        question_lower = question.lower()
        
        if 'why' in question_lower and ('buy' in question_lower or 'sell' in question_lower):
            return ("Trading signals are generated based on technical analysis of multiple indicators "
                   "including moving averages, RSI, MACD, and volume patterns. The model identifies "
                   "potential entry and exit points based on historical patterns and current market conditions. "
                   "However, no signal is guaranteed to be profitable.")
        
        elif 'risk' in question_lower:
            return ("Risk is assessed based on market volatility, position size relative to account balance, "
                   "and technical indicator reliability. Always use proper risk management and never risk "
                   "more than you can afford to lose.")
        
        elif 'trend' in question_lower or 'going' in question_lower:
            trend = market_context.get('trend', 'unclear')
            return (f"Based on technical analysis, the current trend for this asset is {trend}. "
                   "This is determined by analyzing price action, moving averages, and momentum indicators. "
                   "Remember that trends can change quickly in volatile markets.")
        
        else:
            return ("I can provide analysis of trading signals, risk assessments, and market conditions. "
                   "Please ask specific questions about buy/sell signals, risk levels, or market trends. "
                   "Remember that all trading involves risk and past performance is not indicative of future results.")

# Import pandas here to avoid circular imports
import pandas as pd

# Example usage
def main():
    """Example usage of the Chloe LLM module"""
    chloe = ChloeLLM()
    
    # Example technical data
    tech_data = {
        'rsi_14': 65.42,
        'macd': 123.45,
        'macd_signal': 110.23,
        'ema_20': 45678.90,
        'ema_50': 44234.56,
        'volatility': 0.0234
    }
    
    risk_data = {
        'volatility': 0.0234,
        'correlation': 0.65
    }
    
    # Analyze a signal
    analysis = chloe.analyze_signal(
        symbol="BTC/USDT",
        signal="BUY",
        confidence=0.85,
        technical_data=tech_data,
        risk_data=risk_data
    )
    
    print(f"Symbol: {analysis.symbol}")
    print(f"Signal: {analysis.signal}")
    print(f"Confidence: {analysis.confidence}")
    print(f"Explanation: {analysis.explanation}")
    print(f"Suggested Action: {analysis.suggested_action}")
    
    # Generate risk warning
    risk_warning = chloe.generate_risk_warning(
        symbol="BTC/USDT",
        risk_level="MEDIUM",
        risk_factors=["High market volatility", "Elevated RSI levels"]
    )
    print(f"\nRisk Warning:\n{risk_warning}")
    
    print("\nChloe LLM module ready for signal analysis")

if __name__ == "__main__":
    main()