"""
News Sentiment Analysis for Chloe AI 0.4
Professional NLP-based sentiment analysis and news processing system
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import json
import re
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """News article representation"""
    article_id: str
    title: str
    content: str
    source: str
    timestamp: datetime
    url: Optional[str] = None
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []

@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    article_id: str
    compound_score: float  # Overall sentiment (-1 to 1)
    positive_score: float  # Positive sentiment (0 to 1)
    negative_score: float  # Negative sentiment (0 to 1)
    neutral_score: float   # Neutral sentiment (0 to 1)
    confidence: float      # Analysis confidence (0 to 1)
    key_entities: List[str]  # Important entities mentioned
    sentiment_keywords: List[str]  # Keywords driving sentiment

@dataclass
class MarketImpact:
    """Predicted market impact of news"""
    article_id: str
    symbol: str
    price_impact: float    # Expected price impact percentage
    volatility_impact: float  # Expected volatility change
    timeframe: str         # Impact timeframe (SHORT/MEDIUM/LONG)
    confidence: float      # Impact prediction confidence
    drivers: List[str]     # Key factors driving impact

class SentimentAnalyzer:
    """Professional sentiment analysis engine"""
    
    def __init__(self):
        # Financial sentiment lexicon
        self.positive_words = {
            'bullish': 0.8, 'bull': 0.7, 'soar': 0.7, 'surge': 0.7, 'jump': 0.6,
            'gain': 0.6, 'rise': 0.6, 'climb': 0.6, 'boost': 0.6, 'rally': 0.7,
            'outperform': 0.7, 'upgrade': 0.8, 'positive': 0.5, 'strong': 0.5,
            'excellent': 0.8, 'outstanding': 0.9, 'record': 0.7, 'beat': 0.7,
            'profit': 0.6, 'earnings': 0.5, 'growth': 0.6, 'expansion': 0.6
        }
        
        self.negative_words = {
            'bearish': -0.8, 'bear': -0.7, 'crash': -0.9, 'plunge': -0.8, 'drop': -0.6,
            'fall': -0.6, 'decline': -0.6, 'slump': -0.7, 'collapse': -0.9, 'dump': -0.7,
            'sell-off': -0.7, 'downgrade': -0.8, 'negative': -0.5, 'weak': -0.5,
            'disappointing': -0.7, 'miss': -0.7, 'loss': -0.6, 'losses': -0.7,
            'warning': -0.6, 'concern': -0.5, 'risk': -0.5, 'volatile': -0.4
        }
        
        # Intensifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.8, 'significantly': 1.7,
            'massively': 2.2, 'slightly': 0.5, 'marginally': 0.3, 'somewhat': 0.7
        }
        
        # Negation words
        self.negations = {'not', 'no', 'never', 'neither', 'nor', 'nothing', 'nowhere'}
        
        logger.info("🧠 Sentiment Analyzer initialized")

    def analyze_sentiment(self, article: NewsArticle) -> SentimentScore:
        """Analyze sentiment of news article"""
        try:
            text = f"{article.title} {article.content}".lower()
            
            # Tokenize and clean text
            words = self._tokenize_text(text)
            
            # Calculate sentiment scores
            positive_score = 0.0
            negative_score = 0.0
            neutral_score = 0.0
            word_count = 0
            key_entities = []
            sentiment_keywords = []
            
            i = 0
            while i < len(words):
                word = words[i]
                word_count += 1
                
                # Check for intensifiers
                intensity = 1.0
                if i > 0 and words[i-1] in self.intensifiers:
                    intensity = self.intensifiers[words[i-1]]
                
                # Check for negation
                is_negated = i > 0 and words[i-1] in self.negations
                
                # Score positive words
                if word in self.positive_words:
                    score = self.positive_words[word] * intensity
                    if is_negated:
                        negative_score += abs(score)
                        sentiment_keywords.append(f"not_{word}")
                    else:
                        positive_score += score
                        sentiment_keywords.append(word)
                
                # Score negative words
                elif word in self.negative_words:
                    score = abs(self.negative_words[word]) * intensity
                    if is_negated:
                        positive_score += score
                        sentiment_keywords.append(f"not_{word}")
                    else:
                        negative_score += score
                        sentiment_keywords.append(word)
                
                # Extract key entities (symbols, companies, etc.)
                if self._is_financial_entity(word):
                    key_entities.append(word.upper())
                
                i += 1
            
            # Calculate compound score
            total_sentiment = positive_score - negative_score
            compound_score = total_sentiment / max(word_count, 1)
            
            # Normalize scores
            total_magnitude = positive_score + negative_score + 0.1  # Avoid division by zero
            normalized_positive = positive_score / total_magnitude
            normalized_negative = negative_score / total_magnitude
            normalized_neutral = 1.0 - (normalized_positive + normalized_negative)
            
            # Calculate confidence based on sentiment strength
            sentiment_strength = abs(compound_score)
            confidence = min(1.0, sentiment_strength * 2)  # Scale up confidence
            
            return SentimentScore(
                article_id=article.article_id,
                compound_score=max(-1.0, min(1.0, compound_score)),  # Clamp to [-1, 1]
                positive_score=normalized_positive,
                negative_score=normalized_negative,
                neutral_score=max(0.0, normalized_neutral),
                confidence=confidence,
                key_entities=list(set(key_entities)),  # Remove duplicates
                sentiment_keywords=list(set(sentiment_keywords))
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for article {article.article_id}: {e}")
            return self._get_default_sentiment(article.article_id)

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize and clean text"""
        # Remove URLs, special characters
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words and filter
        words = text.lower().split()
        filtered_words = [word.strip() for word in words if len(word) > 2]
        
        return filtered_words

    def _is_financial_entity(self, word: str) -> bool:
        """Check if word is a financial entity"""
        financial_indicators = {
            'btc', 'eth', 'sol', 'ada', 'xrp', 'bnb', 'doge', 'avax', 'matic',
            'bitcoin', 'ethereum', 'solana', 'cardano', 'ripple', 'binance',
            'coinbase', 'tesla', 'apple', 'microsoft', 'google', 'amazon',
            'gold', 'silver', 'oil', 'nasdaq', 'dow', 's&p'
        }
        return word.lower() in financial_indicators

    def _get_default_sentiment(self, article_id: str) -> SentimentScore:
        """Return default sentiment when analysis fails"""
        return SentimentScore(
            article_id=article_id,
            compound_score=0.0,
            positive_score=0.33,
            negative_score=0.33,
            neutral_score=0.34,
            confidence=0.0,
            key_entities=[],
            sentiment_keywords=[]
        )

class NewsImpactPredictor:
    """Predicts market impact of news articles"""
    
    def __init__(self):
        # Impact multipliers by sentiment strength
        self.impact_multipliers = {
            'HIGH': 2.0,
            'MEDIUM': 1.0,
            'LOW': 0.5
        }
        
        # Timeframe impact patterns
        self.timeframe_patterns = {
            'SHORT': {'duration': '1-2 hours', 'volatility': 1.5},
            'MEDIUM': {'duration': '1-3 days', 'volatility': 1.2},
            'LONG': {'duration': '1-2 weeks', 'volatility': 0.8}
        }
        
        logger.info("🔮 News Impact Predictor initialized")

    def predict_impact(self, article: NewsArticle, sentiment: SentimentScore) -> List[MarketImpact]:
        """Predict market impact for relevant symbols"""
        impacts = []
        
        try:
            # Extract symbols from article or use provided ones
            relevant_symbols = article.symbols or sentiment.key_entities
            
            if not relevant_symbols:
                # Try to extract symbols from content
                relevant_symbols = self._extract_symbols_from_content(article)
            
            # Predict impact for each symbol
            for symbol in relevant_symbols:
                impact = self._calculate_single_impact(symbol, sentiment)
                impacts.append(impact)
            
            return impacts
            
        except Exception as e:
            logger.error(f"Impact prediction failed: {e}")
            return []

    def _extract_symbols_from_content(self, article: NewsArticle) -> List[str]:
        """Extract relevant symbols from article content"""
        symbols = []
        content = f"{article.title} {article.content}".upper()
        
        # Common crypto symbols
        crypto_symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'BNB', 'DOGE', 'AVAX', 'MATIC']
        
        for symbol in crypto_symbols:
            if symbol in content:
                symbols.append(f"{symbol}/USDT")
        
        return symbols

    def _calculate_single_impact(self, symbol: str, sentiment: SentimentScore) -> MarketImpact:
        """Calculate impact for single symbol"""
        # Base impact from sentiment strength
        sentiment_strength = abs(sentiment.compound_score)
        base_impact = sentiment.compound_score * 0.02  # ±2% base impact
        
        # Adjust by confidence
        confidence_adjusted_impact = base_impact * sentiment.confidence
        
        # Determine timeframe based on sentiment strength and article characteristics
        if sentiment_strength > 0.6:
            timeframe = 'SHORT'
        elif sentiment_strength > 0.3:
            timeframe = 'MEDIUM'
        else:
            timeframe = 'LONG'
        
        # Calculate volatility impact
        volatility_impact = sentiment_strength * 0.15 * self.timeframe_patterns[timeframe]['volatility']
        
        # Determine impact drivers
        drivers = self._identify_impact_drivers(sentiment)
        
        return MarketImpact(
            article_id=sentiment.article_id,
            symbol=symbol,
            price_impact=confidence_adjusted_impact,
            volatility_impact=volatility_impact,
            timeframe=timeframe,
            confidence=sentiment.confidence,
            drivers=drivers
        )

    def _identify_impact_drivers(self, sentiment: SentimentScore) -> List[str]:
        """Identify key drivers of market impact"""
        drivers = []
        
        # Check for specific keywords
        key_drivers = {
            'regulation': ['regulat', 'ban', 'restrict', 'legal'],
            'adoption': ['adopt', 'integrat', 'partnership', 'collaborat'],
            'technology': ['upgrade', 'launch', 'release', 'develop'],
            'market': ['market', 'trading', 'exchange', 'platform'],
            'macro': ['fed', 'interest', 'inflation', 'economic']
        }
        
        keywords = ' '.join(sentiment.sentiment_keywords).lower()
        
        for driver, terms in key_drivers.items():
            if any(term in keywords for term in terms):
                drivers.append(driver)
        
        return drivers if drivers else ['general_sentiment']

class NewsCollector:
    """Collects news articles from various sources"""
    
    def __init__(self):
        self.collected_articles = deque(maxlen=1000)
        self.last_collection_time = None
        logger.info("📰 News Collector initialized")

    async def collect_news(self, sources: List[str] = None) -> List[NewsArticle]:
        """Collect recent news articles"""
        if sources is None:
            sources = ['crypto_news', 'financial_news', 'market_updates']
        
        articles = []
        
        try:
            # Simulate news collection (would connect to real APIs in production)
            mock_articles = self._generate_mock_news()
            articles.extend(mock_articles)
            
            # Store collected articles
            self.collected_articles.extend(articles)
            self.last_collection_time = datetime.now()
            
            logger.info(f"📰 Collected {len(articles)} news articles")
            return articles
            
        except Exception as e:
            logger.error(f"News collection failed: {e}")
            return []

    def _generate_mock_news(self) -> List[NewsArticle]:
        """Generate mock news articles for demonstration"""
        mock_news = [
            NewsArticle(
                article_id="news_001",
                title="Bitcoin Surges 5% After Major Institutional Adoption",
                content="Major pension fund announces $1 billion Bitcoin investment, driving prices higher. Analysts predict continued bullish momentum.",
                source="CryptoDaily",
                timestamp=datetime.now() - timedelta(hours=1),
                symbols=["BTC/USDT"]
            ),
            NewsArticle(
                article_id="news_002",
                title="Ethereum Developers Announce Major Scaling Upgrade",
                content="Ethereum core developers reveal breakthrough scaling solution that could reduce gas fees by 80%. Network upgrade scheduled for next month.",
                source="ETHNews",
                timestamp=datetime.now() - timedelta(hours=2),
                symbols=["ETH/USDT"]
            ),
            NewsArticle(
                article_id="news_003",
                title="Regulatory Concerns Cause Crypto Market Volatility",
                content="New regulatory proposals from financial authorities create uncertainty in cryptocurrency markets. Traders advised to exercise caution.",
                source="FinancialTimes",
                timestamp=datetime.now() - timedelta(hours=3),
                symbols=["BTC/USDT", "ETH/USDT"]
            ),
            NewsArticle(
                article_id="news_004",
                title="Solana Ecosystem Sees Record DeFi Activity",
                content="Decentralized finance protocols on Solana blockchain report unprecedented user growth and transaction volume. SOL token gaining attention.",
                source="DeFiWatch",
                timestamp=datetime.now() - timedelta(hours=4),
                symbols=["SOL/USDT"]
            )
        ]
        
        return mock_news

class NewsSentimentSystem:
    """Main news sentiment analysis system"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.impact_predictor = NewsImpactPredictor()
        self.news_collector = NewsCollector()
        
        # Storage
        self.processed_articles = deque(maxlen=1000)
        self.sentiment_history = deque(maxlen=1000)
        self.impact_predictions = deque(maxlen=1000)
        
        logger.info("🗞️ News Sentiment System initialized")

    async def process_news_cycle(self) -> Dict:
        """Process complete news analysis cycle"""
        try:
            logger.info("🔄 Starting News Sentiment Analysis Cycle...")
            
            # Step 1: Collect news
            logger.info("   1️⃣ Collecting news articles...")
            articles = await self.news_collector.collect_news()
            logger.info(f"      Collected {len(articles)} articles")
            
            if not articles:
                return {"status": "NO_NEWS", "message": "No news articles collected"}
            
            # Step 2: Analyze sentiment
            logger.info("   2️⃣ Analyzing sentiment...")
            sentiment_results = []
            
            for article in articles:
                sentiment = self.sentiment_analyzer.analyze_sentiment(article)
                sentiment_results.append(sentiment)
                
                # Store results
                self.processed_articles.append(article)
                self.sentiment_history.append(sentiment)
            
            logger.info(f"      Analyzed {len(sentiment_results)} articles")
            
            # Step 3: Predict market impact
            logger.info("   3️⃣ Predicting market impact...")
            impact_results = []
            
            for article, sentiment in zip(articles, sentiment_results):
                impacts = self.impact_predictor.predict_impact(article, sentiment)
                impact_results.extend(impacts)
                
                # Store impacts
                self.impact_predictions.extend(impacts)
            
            logger.info(f"      Generated {len(impact_results)} impact predictions")
            
            # Step 4: Generate summary
            summary = self._generate_analysis_summary(articles, sentiment_results, impact_results)
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "status": "SUCCESS",
                "articles_processed": len(articles),
                "sentiment_analyses": len(sentiment_results),
                "impact_predictions": len(impact_results),
                "summary": summary,
                "recent_sentiments": [self._serialize_sentiment(s) for s in list(self.sentiment_history)[-5:]],
                "recent_impacts": [self._serialize_impact(i) for i in list(self.impact_predictions)[-5:]]
            }
            
            logger.info("✅ News sentiment analysis cycle completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ News analysis cycle failed: {e}")
            return {
                "status": "ERROR",
                "error": str(e)
            }

    def _generate_analysis_summary(self, articles: List[NewsArticle], 
                                 sentiments: List[SentimentScore],
                                 impacts: List[MarketImpact]) -> Dict:
        """Generate analysis summary"""
        # Calculate overall sentiment
        avg_compound = np.mean([s.compound_score for s in sentiments])
        avg_confidence = np.mean([s.confidence for s in sentiments])
        
        # Count sentiment distribution
        positive_count = sum(1 for s in sentiments if s.compound_score > 0.1)
        negative_count = sum(1 for s in sentiments if s.compound_score < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Extract key symbols
        all_symbols = []
        for impact in impacts:
            all_symbols.append(impact.symbol)
        unique_symbols = list(set(all_symbols))
        
        # Calculate expected market movements
        symbol_impacts = {}
        for symbol in unique_symbols:
            symbol_impacts[symbol] = {
                'avg_price_impact': np.mean([i.price_impact for i in impacts if i.symbol == symbol]),
                'total_volatility_impact': sum([i.volatility_impact for i in impacts if i.symbol == symbol]),
                'mentions': sum(1 for i in impacts if i.symbol == symbol)
            }
        
        return {
            "overall_sentiment": avg_compound,
            "sentiment_confidence": avg_confidence,
            "distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count
            },
            "covered_symbols": unique_symbols,
            "symbol_impacts": symbol_impacts,
            "total_impact_predictions": len(impacts)
        }

    def _serialize_sentiment(self, sentiment: SentimentScore) -> Dict:
        """Serialize sentiment for JSON output"""
        return {
            "article_id": sentiment.article_id,
            "compound_score": sentiment.compound_score,
            "confidence": sentiment.confidence,
            "key_entities": sentiment.key_entities
        }

    def _serialize_impact(self, impact: MarketImpact) -> Dict:
        """Serialize impact for JSON output"""
        return {
            "article_id": impact.article_id,
            "symbol": impact.symbol,
            "price_impact": impact.price_impact,
            "volatility_impact": impact.volatility_impact,
            "timeframe": impact.timeframe,
            "confidence": impact.confidence
        }

# Global news sentiment system instance
news_system = None

def get_news_sentiment_system() -> NewsSentimentSystem:
    """Get singleton news sentiment system instance"""
    global news_system
    if news_system is None:
        news_system = NewsSentimentSystem()
    return news_system

def main():
    """Example usage"""
    print("🗞️ News Sentiment Analysis System ready")

if __name__ == "__main__":
    main()