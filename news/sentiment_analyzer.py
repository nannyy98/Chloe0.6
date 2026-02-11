"""
Market Sentiment Analyzer for Chloe AI
Analyzes news sentiment and integrates with trading signals
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
import json

# Import our modules
from news.news_collector import NewsCollector, NewsArticle
from news.nlp_processor import NLPProcessor, AdvancedSentimentAnalyzer, SentimentResult, TopicResult

logger = logging.getLogger(__name__)

@dataclass
class MarketSentiment:
    """Market sentiment analysis result"""
    symbol: str
    overall_sentiment: float  # -1 to 1
    sentiment_trend: str     # BULLISH, BEARISH, NEUTRAL, MIXED
    confidence: float        # 0 to 1
    volume: int             # Number of articles analyzed
    recent_impact: float    # Impact of recent news
    topics: Dict[str, float] # Topic weights
    sources: Dict[str, int]  # Source distribution
    timestamp: datetime

@dataclass
class SentimentSignal:
    """Trading signal based on sentiment analysis"""
    symbol: str
    signal: str             # BUY, SELL, HOLD
    strength: float         # 0 to 1
    confidence: float       # 0 to 1
    sentiment_score: float  # -1 to 1
    reasoning: str          # Explanation
    timestamp: datetime

class SentimentAnalyzer:
    """
    Main sentiment analyzer that integrates news collection and NLP processing
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['BTC', 'ETH', 'ADA', 'BNB']
        self.news_collector = NewsCollector()
        self.nlp_processor = NLPProcessor()
        self.advanced_analyzer = AdvancedSentimentAnalyzer()
        
        # Sentiment buffers for trend analysis
        self.sentiment_history = {symbol: deque(maxlen=100) for symbol in self.symbols}
        self.article_history = {symbol: deque(maxlen=200) for symbol in self.symbols}
        
        # Configuration
        self.sentiment_thresholds = {
            'strong_buy': 0.4,
            'buy': 0.2,
            'sell': -0.2,
            'strong_sell': -0.4
        }
        
        self.time_weights = {
            'hour': 1.0,      # Last hour - highest weight
            '3_hours': 0.8,   # Last 3 hours
            '12_hours': 0.6,  # Last 12 hours
            '24_hours': 0.4   # Last 24 hours
        }
        
        # Impact multipliers
        self.source_multipliers = {
            'news': 1.0,
            'social_media': 0.7,
            'reddit': 0.8,
            'twitter': 0.6
        }
        
        self.topic_multipliers = {
            'regulation': 1.5,
            'security': 1.3,
            'adoption': 1.2,
            'technology': 1.1,
            'market': 1.0,
            'macro': 1.4
        }
        
        logger.info("📊 Market Sentiment Analyzer initialized")
    
    async def analyze_market_sentiment(self, symbol: str, hours: int = 24) -> MarketSentiment:
        """
        Analyze overall market sentiment for a symbol
        
        Args:
            symbol: Trading symbol to analyze
            hours: Time period to analyze
            
        Returns:
            MarketSentiment object with analysis results
        """
        logger.info(f"📊 Analyzing market sentiment for {symbol} over {hours} hours...")
        
        try:
            # Collect recent news
            async with self.news_collector:
                all_articles = await self.news_collector.collect_all_news()
            
            # Filter articles for the symbol
            symbol_articles = [article for article in all_articles 
                             if symbol.upper() in [s.upper() for s in article.symbols]]
            
            # Filter by timeframe
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_articles = [article for article in symbol_articles 
                             if article.timestamp >= cutoff_time]
            
            if not recent_articles:
                logger.warning(f"⚠️ No recent articles found for {symbol}")
                return self._create_neutral_sentiment(symbol)
            
            # Process articles with NLP
            processed_articles = []
            for article in recent_articles:
                # Analyze sentiment
                sentiment = self.advanced_analyzer.analyze_with_context(
                    article.content, 
                    self._get_source_type(article.source)
                )
                
                # Extract topics
                topics = self.nlp_processor.extract_topics(article.content)
                
                # Calculate market impact
                impact = self.nlp_processor.calculate_market_impact(
                    sentiment, 
                    topics, 
                    self._calculate_relevance(article, symbol)
                )
                
                processed_articles.append({
                    'article': article,
                    'sentiment': sentiment,
                    'topics': topics,
                    'impact': impact
                })
            
            # Calculate overall sentiment
            market_sentiment = self._calculate_overall_sentiment(symbol, processed_articles)
            
            # Store in history
            self.sentiment_history[symbol].append(market_sentiment)
            for article_data in processed_articles:
                self.article_history[symbol].append(article_data)
            
            logger.info(f"✅ Sentiment analysis completed for {symbol}")
            return market_sentiment
            
        except Exception as e:
            logger.error(f"❌ Error analyzing sentiment for {symbol}: {e}")
            return self._create_neutral_sentiment(symbol)
    
    def _calculate_overall_sentiment(self, symbol: str, 
                                   processed_articles: List[Dict]) -> MarketSentiment:
        """Calculate overall sentiment from processed articles"""
        if not processed_articles:
            return self._create_neutral_sentiment(symbol)
        
        # Weighted sentiment calculation
        weighted_sentiments = []
        total_weight = 0
        topics_counter = defaultdict(float)
        sources_counter = defaultdict(int)
        
        current_time = datetime.now()
        
        for article_data in processed_articles:
            article = article_data['article']
            sentiment = article_data['sentiment']
            topics = article_data['topics']
            impact = article_data['impact']
            
            # Calculate time weight
            time_diff = (current_time - article.timestamp).total_seconds() / 3600  # hours
            if time_diff <= 1:
                time_weight = self.time_weights['hour']
            elif time_diff <= 3:
                time_weight = self.time_weights['3_hours']
            elif time_diff <= 12:
                time_weight = self.time_weights['12_hours']
            else:
                time_weight = self.time_weights['24_hours']
            
            # Calculate source weight
            source_type = self._get_source_type(article.source)
            source_weight = self.source_multipliers.get(source_type, 0.5)
            
            # Calculate topic weights
            topic_weight = 1.0
            for topic in topics.topics:
                topic_weight *= self.topic_multipliers.get(topic, 1.0)
            
            # Final weight
            final_weight = time_weight * source_weight * topic_weight
            weighted_sentiment = impact * final_weight * sentiment.confidence
            
            weighted_sentiments.append(weighted_sentiment)
            total_weight += final_weight
            
            # Track topics
            for topic, score in topics.topic_scores.items():
                topics_counter[topic] += score * final_weight
            
            # Track sources
            sources_counter[article.source] += 1
        
        # Calculate final sentiment
        if total_weight > 0:
            overall_sentiment = sum(weighted_sentiments) / total_weight
            confidence = min(1.0, total_weight / len(processed_articles))
        else:
            overall_sentiment = 0.0
            confidence = 0.0
        
        # Determine trend
        sentiment_trend = self._determine_sentiment_trend(overall_sentiment)
        
        # Calculate recent impact (last hour)
        recent_impact = self._calculate_recent_impact(processed_articles)
        
        # Normalize topic weights
        total_topic_score = sum(topics_counter.values())
        if total_topic_score > 0:
            normalized_topics = {topic: score/total_topic_score 
                               for topic, score in topics_counter.items()}
        else:
            normalized_topics = {}
        
        return MarketSentiment(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            sentiment_trend=sentiment_trend,
            confidence=confidence,
            volume=len(processed_articles),
            recent_impact=recent_impact,
            topics=normalized_topics,
            sources=dict(sources_counter),
            timestamp=datetime.now()
        )
    
    def _create_neutral_sentiment(self, symbol: str) -> MarketSentiment:
        """Create neutral sentiment result when no data available"""
        return MarketSentiment(
            symbol=symbol,
            overall_sentiment=0.0,
            sentiment_trend='NEUTRAL',
            confidence=0.0,
            volume=0,
            recent_impact=0.0,
            topics={},
            sources={},
            timestamp=datetime.now()
        )
    
    def _get_source_type(self, source: str) -> str:
        """Determine source type for weighting"""
        source_lower = source.lower()
        if 'reddit' in source_lower:
            return 'reddit'
        elif 'twitter' in source_lower or 'x.com' in source_lower:
            return 'twitter'
        elif any(social in source_lower for social in ['facebook', 'telegram', 'discord']):
            return 'social_media'
        else:
            return 'news'
    
    def _calculate_relevance(self, article: NewsArticle, symbol: str) -> float:
        """Calculate relevance of article to specific symbol"""
        relevance = 0.5  # Base relevance
        
        # Higher relevance if symbol is mentioned multiple times
        symbol_mentions = article.content.upper().count(symbol.upper())
        relevance += min(0.3, symbol_mentions * 0.1)
        
        # Higher relevance for specific symbol news vs general crypto news
        if symbol.upper() in ['BTC', 'ETH'] and len(article.symbols) == 1:
            relevance += 0.2  # Specific major coin news
        
        return min(1.0, relevance)
    
    def _determine_sentiment_trend(self, sentiment_score: float) -> str:
        """Determine sentiment trend based on score"""
        if sentiment_score > 0.3:
            return 'BULLISH'
        elif sentiment_score > 0.1:
            return 'MODERATELY_BULLISH'
        elif sentiment_score < -0.3:
            return 'BEARISH'
        elif sentiment_score < -0.1:
            return 'MODERATELY_BEARISH'
        else:
            return 'NEUTRAL'
    
    def _calculate_recent_impact(self, processed_articles: List[Dict]) -> float:
        """Calculate impact of recent news (last hour)"""
        recent_articles = []
        current_time = datetime.now()
        
        for article_data in processed_articles:
            article = article_data['article']
            if (current_time - article.timestamp).total_seconds() <= 3600:  # Last hour
                recent_articles.append(article_data['impact'] * article_data['sentiment'].confidence)
        
        if recent_articles:
            return np.mean(recent_articles)
        return 0.0
    
    def generate_sentiment_signal(self, symbol: str, 
                                market_sentiment: MarketSentiment) -> SentimentSignal:
        """
        Generate trading signal based on sentiment analysis
        
        Args:
            symbol: Trading symbol
            market_sentiment: Market sentiment analysis
            
        Returns:
            SentimentSignal object
        """
        sentiment_score = market_sentiment.overall_sentiment
        confidence = market_sentiment.confidence
        
        # Determine signal based on thresholds
        if sentiment_score >= self.sentiment_thresholds['strong_buy'] and confidence > 0.7:
            signal = 'STRONG_BUY'
            strength = min(1.0, (sentiment_score - self.sentiment_thresholds['strong_buy']) / 0.2 + 0.7)
        elif sentiment_score >= self.sentiment_thresholds['buy'] and confidence > 0.5:
            signal = 'BUY'
            strength = min(1.0, (sentiment_score - self.sentiment_thresholds['buy']) / 0.2 + 0.5)
        elif sentiment_score <= self.sentiment_thresholds['strong_sell'] and confidence > 0.7:
            signal = 'STRONG_SELL'
            strength = min(1.0, (abs(sentiment_score) - abs(self.sentiment_thresholds['strong_sell'])) / 0.2 + 0.7)
        elif sentiment_score <= self.sentiment_thresholds['sell'] and confidence > 0.5:
            signal = 'SELL'
            strength = min(1.0, (abs(sentiment_score) - abs(self.sentiment_thresholds['sell'])) / 0.2 + 0.5)
        else:
            signal = 'HOLD'
            strength = 1.0 - abs(sentiment_score)  # Higher strength for neutral sentiment
        
        # Generate reasoning
        reasoning = self._generate_reasoning(symbol, market_sentiment, signal)
        
        return SentimentSignal(
            symbol=symbol,
            signal=signal,
            strength=strength,
            confidence=confidence,
            sentiment_score=sentiment_score,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def _generate_reasoning(self, symbol: str, sentiment: MarketSentiment, signal: str) -> str:
        """Generate human-readable reasoning for sentiment signal"""
        base_reasoning = f"Based on analysis of {sentiment.volume} articles over 24 hours. "
        
        # Add sentiment description
        if sentiment.sentiment_trend == 'BULLISH':
            trend_desc = "Strongly positive sentiment with bullish market trend."
        elif sentiment.sentiment_trend == 'MODERATELY_BULLISH':
            trend_desc = "Moderately positive sentiment with upward trend."
        elif sentiment.sentiment_trend == 'BEARISH':
            trend_desc = "Strongly negative sentiment with bearish market trend."
        elif sentiment.sentiment_trend == 'MODERATELY_BEARISH':
            trend_desc = "Moderately negative sentiment with downward trend."
        else:
            trend_desc = "Neutral sentiment with mixed market signals."
        
        # Add topic insights
        if sentiment.topics:
            top_topics = sorted(sentiment.topics.items(), key=lambda x: x[1], reverse=True)[:2]
            topic_desc = f" Key topics include: {', '.join([topic for topic, _ in top_topics])}."
        else:
            topic_desc = " No dominant topics identified."
        
        # Add source distribution
        if sentiment.sources:
            top_sources = sorted(sentiment.sources.items(), key=lambda x: x[1], reverse=True)[:2]
            source_desc = f" Primary sources: {', '.join([source for source, _ in top_sources])}."
        else:
            source_desc = ""
        
        return base_reasoning + trend_desc + topic_desc + source_desc
    
    def get_portfolio_sentiment(self) -> Dict[str, MarketSentiment]:
        """Get sentiment analysis for all monitored symbols"""
        portfolio_sentiment = {}
        
        for symbol in self.symbols:
            sentiment = self.analyze_market_sentiment(symbol)
            portfolio_sentiment[symbol] = sentiment
        
        return portfolio_sentiment
    
    def get_sentiment_history(self, symbol: str, hours: int = 24) -> List[MarketSentiment]:
        """Get sentiment history for trend analysis"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [sentiment for sentiment in self.sentiment_history.get(symbol, [])
                if sentiment.timestamp >= cutoff_time]
    
    def integrate_with_trading_signals(self, sentiment_signal: SentimentSignal,
                                     technical_signal: str = 'HOLD',
                                     technical_confidence: float = 0.5) -> Dict:
        """
        Integrate sentiment signal with technical analysis signals
        
        Args:
            sentiment_signal: Sentiment-based signal
            technical_signal: Technical analysis signal
            technical_confidence: Confidence in technical signal
            
        Returns:
            Integrated signal with reasoning
        """
        # Weight sentiment vs technical analysis
        sentiment_weight = 0.4
        technical_weight = 0.6
        
        # Convert signals to numerical values
        signal_mapping = {
            'STRONG_BUY': 2,
            'BUY': 1,
            'HOLD': 0,
            'SELL': -1,
            'STRONG_SELL': -2
        }
        
        sentiment_value = signal_mapping.get(sentiment_signal.signal, 0)
        technical_value = signal_mapping.get(technical_signal, 0)
        
        # Calculate weighted combined signal
        combined_value = (sentiment_value * sentiment_signal.confidence * sentiment_weight +
                         technical_value * technical_confidence * technical_weight)
        
        # Determine final signal
        if combined_value >= 1.5:
            final_signal = 'STRONG_BUY'
        elif combined_value >= 0.5:
            final_signal = 'BUY'
        elif combined_value <= -1.5:
            final_signal = 'STRONG_SELL'
        elif combined_value <= -0.5:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        # Calculate combined confidence
        combined_confidence = (sentiment_signal.confidence * sentiment_weight +
                              technical_confidence * technical_weight)
        
        # Generate integration reasoning
        reasoning = (
            f"Combined signal from sentiment analysis ({sentiment_signal.signal}) "
            f"and technical analysis ({technical_signal}). "
            f"Sentiment confidence: {sentiment_signal.confidence:.2f}, "
            f"Technical confidence: {technical_confidence:.2f}"
        )
        
        return {
            'signal': final_signal,
            'confidence': combined_confidence,
            'reasoning': reasoning,
            'sentiment_signal': sentiment_signal.signal,
            'technical_signal': technical_signal,
            'combined_value': combined_value
        }

# Example usage
async def main():
    """Example usage of Sentiment Analyzer"""
    analyzer = SentimentAnalyzer(['BTC', 'ETH'])
    
    # Analyze sentiment for Bitcoin
    print("📊 Analyzing Bitcoin sentiment...")
    btc_sentiment = await analyzer.analyze_market_sentiment('BTC', hours=24)
    
    print(f"Symbol: {btc_sentiment.symbol}")
    print(f"Overall Sentiment: {btc_sentiment.overall_sentiment:.3f}")
    print(f"Sentiment Trend: {btc_sentiment.sentiment_trend}")
    print(f"Confidence: {btc_sentiment.confidence:.3f}")
    print(f"Volume: {btc_sentiment.volume} articles")
    print(f"Topics: {btc_sentiment.topics}")
    print(f"Sources: {btc_sentiment.sources}")
    
    # Generate trading signal
    sentiment_signal = analyzer.generate_sentiment_signal('BTC', btc_sentiment)
    print(f"\n Trading Signal: {sentiment_signal.signal}")
    print(f"Strength: {sentiment_signal.strength:.3f}")
    print(f"Confidence: {sentiment_signal.confidence:.3f}")
    print(f"Reasoning: {sentiment_signal.reasoning}")
    
    # Integration with technical analysis
    integrated_signal = analyzer.integrate_with_trading_signals(
        sentiment_signal, 
        'BUY', 
        0.7
    )
    
    print(f"\nIntegrated Signal: {integrated_signal['signal']}")
    print(f"Combined Confidence: {integrated_signal['confidence']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())