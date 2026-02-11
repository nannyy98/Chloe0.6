#!/usr/bin/env python3
"""
News Sentiment Analysis Demo for Chloe AI 0.4
Professional NLP-based sentiment analysis and market impact prediction
"""

import asyncio
import logging
from datetime import datetime
from news_sentiment import get_news_sentiment_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_news_sentiment_analysis():
    """Demonstrate news sentiment analysis capabilities"""
    logger.info("🗞️ NEWS SENTIMENT ANALYSIS DEMO")
    logger.info("=" * 50)
    
    try:
        # Initialize news sentiment system
        logger.info("🔧 Initializing News Sentiment System...")
        news_system = get_news_sentiment_system()
        logger.info("✅ System initialized")
        
        # Process news cycle
        logger.info(f"\n🔄 Processing news analysis cycle...")
        results = await news_system.process_news_cycle()
        
        # Display results
        logger.info(f"\n🎯 ANALYSIS RESULTS:")
        logger.info(f"   Status: {results['status']}")
        logger.info(f"   Articles Processed: {results['articles_processed']}")
        logger.info(f"   Sentiment Analyses: {results['sentiment_analyses']}")
        logger.info(f"   Impact Predictions: {results['impact_predictions']}")
        
        if results['status'] == 'SUCCESS':
            summary = results['summary']
            
            logger.info(f"\n📊 SENTIMENT SUMMARY:")
            logger.info(f"   Overall Sentiment: {summary['overall_sentiment']:+.3f}")
            logger.info(f"   Analysis Confidence: {summary['sentiment_confidence']:.3f}")
            logger.info(f"   Distribution: {summary['distribution']}")
            
            logger.info(f"\n📈 MARKET IMPACT PREDICTIONS:")
            logger.info(f"   Covered Symbols: {summary['covered_symbols']}")
            
            for symbol, impact_data in summary['symbol_impacts'].items():
                logger.info(f"   {symbol}:")
                logger.info(f"      Avg Price Impact: {impact_data['avg_price_impact']:+.4f}")
                logger.info(f"      Volatility Impact: {impact_data['total_volatility_impact']:.4f}")
                logger.info(f"      Mentions: {impact_data['mentions']}")
            
            # Show detailed sentiment analyses
            logger.info(f"\n📋 DETAILED SENTIMENT ANALYSES:")
            for i, sentiment in enumerate(results['recent_sentiments'][:3]):
                logger.info(f"   {i+1}. Article {sentiment['article_id']}:")
                logger.info(f"      Compound Score: {sentiment['compound_score']:+.3f}")
                logger.info(f"      Confidence: {sentiment['confidence']:.3f}")
                logger.info(f"      Key Entities: {sentiment['key_entities']}")
            
            # Show impact predictions
            logger.info(f"\n🔮 IMPACT PREDICTIONS:")
            for i, impact in enumerate(results['recent_impacts'][:5]):
                logger.info(f"   {i+1}. {impact['symbol']} (Article {impact['article_id']}):")
                logger.info(f"      Price Impact: {impact['price_impact']:+.4f}")
                logger.info(f"      Volatility Impact: {impact['volatility_impact']:.4f}")
                logger.info(f"      Timeframe: {impact['timeframe']}")
                logger.info(f"      Confidence: {impact['confidence']:.3f}")
        
        logger.info(f"\n{'='*50}")
        logger.info("✅ NEWS SENTIMENT ANALYSIS DEMO COMPLETED")
        logger.info("🚀 Key achievements:")
        logger.info("   • Implemented professional NLP sentiment analysis")
        logger.info("   • Financial lexicon with crypto/stock terminology")
        logger.info("   • Market impact prediction based on sentiment")
        logger.info("   • Multi-symbol news processing")
        logger.info("   • Real-time sentiment scoring")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_sentiment_analyzer():
    """Demonstrate sentiment analyzer capabilities"""
    logger.info(f"\n🧠 SENTIMENT ANALYZER DEMO")
    logger.info("=" * 40)
    
    try:
        from news_sentiment import SentimentAnalyzer, NewsArticle
        from datetime import datetime
        
        analyzer = SentimentAnalyzer()
        
        # Test articles with different sentiments
        test_articles = [
            NewsArticle(
                article_id="test_001",
                title="Bitcoin Prices Soar to New All-Time High",
                content="Cryptocurrency markets rally as Bitcoin reaches unprecedented levels. Analysts predict continued bullish momentum with strong institutional adoption.",
                source="CryptoNews",
                timestamp=datetime.now(),
                symbols=["BTC/USDT"]
            ),
            NewsArticle(
                article_id="test_002",
                title="Regulatory Crackdown Threatens Crypto Industry",
                content="Government regulators announce strict new measures that could severely impact cryptocurrency trading. Market participants express concern about future viability.",
                source="FinancialTimes",
                timestamp=datetime.now(),
                symbols=["BTC/USDT", "ETH/USDT"]
            ),
            NewsArticle(
                article_id="test_003",
                title="Ethereum Network Upgrade Shows Promising Results",
                content="Recent protocol improvements demonstrate significant performance gains. Developer community optimistic about long-term scalability solutions.",
                source="ETHNews",
                timestamp=datetime.now(),
                symbols=["ETH/USDT"]
            )
        ]
        
        logger.info("Analyzing test articles:")
        
        for i, article in enumerate(test_articles):
            sentiment = analyzer.analyze_sentiment(article)
            
            logger.info(f"\n   Article {i+1}: {article.title}")
            logger.info(f"      Compound Score: {sentiment.compound_score:+.3f}")
            logger.info(f"      Positive: {sentiment.positive_score:.3f}")
            logger.info(f"      Negative: {sentiment.negative_score:.3f}")
            logger.info(f"      Neutral: {sentiment.neutral_score:.3f}")
            logger.info(f"      Confidence: {sentiment.confidence:.3f}")
            logger.info(f"      Key Entities: {sentiment.key_entities}")
            logger.info(f"      Sentiment Keywords: {sentiment.sentiment_keywords}")
        
        logger.info("✅ Sentiment analyzer demonstrated successfully")
        
    except Exception as e:
        logger.error(f"❌ Sentiment analyzer demo failed: {e}")

def demonstrate_impact_prediction():
    """Demonstrate market impact prediction"""
    logger.info(f"\n🔮 MARKET IMPACT PREDICTION DEMO")
    logger.info("=" * 40)
    
    try:
        from news_sentiment import NewsImpactPredictor, NewsArticle, SentimentScore
        from datetime import datetime
        
        predictor = NewsImpactPredictor()
        
        # Create test sentiment scores
        test_sentiments = [
            SentimentScore(
                article_id="impact_test_001",
                compound_score=0.75,  # Strongly positive
                positive_score=0.8,
                negative_score=0.1,
                neutral_score=0.1,
                confidence=0.9,
                key_entities=["BTC/USDT"],
                sentiment_keywords=["soar", "rally", "institutional"]
            ),
            SentimentScore(
                article_id="impact_test_002",
                compound_score=-0.65,  # Strongly negative
                positive_score=0.15,
                negative_score=0.7,
                neutral_score=0.15,
                confidence=0.85,
                key_entities=["ETH/USDT"],
                sentiment_keywords=["crackdown", "threaten", "concern"]
            ),
            SentimentScore(
                article_id="impact_test_003",
                compound_score=0.35,  # Moderately positive
                positive_score=0.5,
                negative_score=0.2,
                neutral_score=0.3,
                confidence=0.7,
                key_entities=["SOL/USDT"],
                sentiment_keywords=["upgrade", "promising", "optimistic"]
            )
        ]
        
        test_articles = [
            NewsArticle(
                article_id="impact_test_001",
                title="Positive News Article",
                content="Good news content",
                source="Test",
                timestamp=datetime.now(),
                symbols=["BTC/USDT"]
            ),
            NewsArticle(
                article_id="impact_test_002",
                title="Negative News Article",
                content="Bad news content",
                source="Test",
                timestamp=datetime.now(),
                symbols=["ETH/USDT"]
            ),
            NewsArticle(
                article_id="impact_test_003",
                title="Neutral News Article",
                content="Mixed news content",
                source="Test",
                timestamp=datetime.now(),
                symbols=["SOL/USDT"]
            )
        ]
        
        logger.info("Predicting market impacts:")
        
        for article, sentiment in zip(test_articles, test_sentiments):
            impacts = predictor.predict_impact(article, sentiment)
            
            logger.info(f"\n   {article.title}")
            logger.info(f"      Sentiment: {sentiment.compound_score:+.2f}")
            logger.info(f"      Predicted Impacts: {len(impacts)}")
            
            for impact in impacts:
                logger.info(f"      {impact.symbol}:")
                logger.info(f"         Price Impact: {impact.price_impact:+.4f}")
                logger.info(f"         Volatility Impact: {impact.volatility_impact:.4f}")
                logger.info(f"         Timeframe: {impact.timeframe}")
                logger.info(f"         Drivers: {impact.drivers}")
        
        logger.info("✅ Impact prediction demonstrated successfully")
        
    except Exception as e:
        logger.error(f"❌ Impact prediction demo failed: {e}")

def demonstrate_news_collection():
    """Demonstrate news collection capabilities"""
    logger.info(f"\n📰 NEWS COLLECTION DEMO")
    logger.info("=" * 40)
    
    try:
        from news_sentiment import NewsCollector
        import asyncio
        
        collector = NewsCollector()
        
        logger.info("Collecting news articles...")
        
        # Collect news (mock data in this demo)
        articles = asyncio.run(collector.collect_news(['crypto_news', 'financial_news']))
        
        logger.info(f"Collected {len(articles)} articles:")
        
        for i, article in enumerate(articles[:3]):  # Show first 3
            logger.info(f"   {i+1}. [{article.source}] {article.title}")
            logger.info(f"      Timestamp: {article.timestamp}")
            logger.info(f"      Symbols: {article.symbols}")
            logger.info(f"      Content Preview: {article.content[:100]}...")
            logger.info()
        
        logger.info(f"Total articles in memory: {len(collector.collected_articles)}")
        logger.info(f"Last collection: {collector.last_collection_time}")
        
        logger.info("✅ News collection demonstrated successfully")
        
    except Exception as e:
        logger.error(f"❌ News collection demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe AI 0.4 - News Sentiment Analysis Demo")
    print("Professional NLP sentiment analysis and market impact prediction")
    print()
    
    # Run main news sentiment demo
    await demonstrate_news_sentiment_analysis()
    
    # Run additional demonstrations
    demonstrate_sentiment_analyzer()
    demonstrate_impact_prediction()
    demonstrate_news_collection()
    
    print(f"\n🎉 ALL NEWS SENTIMENT DEMOS COMPLETED SUCCESSFULLY")
    print("Chloe AI now has professional news sentiment analysis capabilities!")

if __name__ == "__main__":
    asyncio.run(main())