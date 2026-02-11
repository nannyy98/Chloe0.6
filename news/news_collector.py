"""
News Collector for Chloe AI
Collects news and social media data from multiple sources for sentiment analysis
"""

import asyncio
import aiohttp
import feedparser
import logging
import json
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from urllib.parse import quote
import re

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    symbols: List[str]
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    topics: List[str] = None

class NewsCollector:
    """
    Collects news from multiple sources including RSS feeds, APIs, and social media
    """
    
    def __init__(self):
        self.session = None
        self.collected_articles = []
        self.symbols = ['BTC', 'ETH', 'ADA', 'BNB', 'SOL', 'XRP', 'DOT', 'DOGE']
        self.crypto_keywords = [
            'bitcoin', 'ethereum', 'cryptocurrency', 'crypto', 'blockchain',
            'defi', 'nft', 'web3', 'token', 'coin', 'altcoin'
        ]
        
        # RSS feeds configuration
        self.rss_feeds = {
            'coindesk': 'https://feeds.feedburner.com/CoinDesk',
            'cointelegraph': 'https://cointelegraph.com/rss',
            'bitcoin_news': 'https://news.bitcoin.com/feed/',
            'cryptonews': 'https://cryptonews.com/news/rss/',
            'reuters_crypto': 'https://www.reutersagency.com/feed/?best-topics=business-finance',
            'cnbc_crypto': 'https://www.cnbc.com/id/10000739/device/rss/rss.html'
        }
        
        # API configurations
        self.news_api_key = None  # Would be set from config
        self.reddit_client_id = None
        self.reddit_client_secret = None
        
        # Collection settings
        self.max_articles_per_source = 50
        self.article_age_limit = timedelta(hours=24)  # Only collect recent articles
        self.min_content_length = 100  # Minimum content length for analysis
        
        logger.info("🗞️ News Collector initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def collect_all_news(self) -> List[NewsArticle]:
        """Collect news from all sources"""
        logger.info("🗞️ Starting comprehensive news collection...")
        
        all_articles = []
        
        # Collect from RSS feeds
        rss_articles = await self._collect_rss_feeds()
        all_articles.extend(rss_articles)
        
        # Collect from News APIs (if configured)
        if self.news_api_key:
            api_articles = await self._collect_news_api()
            all_articles.extend(api_articles)
        
        # Collect from social media (Reddit, Twitter)
        social_articles = await self._collect_social_media()
        all_articles.extend(social_articles)
        
        # Process and deduplicate
        processed_articles = self._process_articles(all_articles)
        self.collected_articles = processed_articles
        
        logger.info(f"✅ Collected {len(processed_articles)} unique articles")
        return processed_articles
    
    async def _collect_rss_feeds(self) -> List[NewsArticle]:
        """Collect articles from RSS feeds"""
        articles = []
        logger.info("📡 Collecting from RSS feeds...")
        
        async def fetch_feed(source_name: str, url: str):
            try:
                # Using feedparser (synchronous) in async context
                loop = asyncio.get_event_loop()
                feed = await loop.run_in_executor(None, feedparser.parse, url)
                
                source_articles = []
                for entry in feed.entries[:self.max_articles_per_source]:
                    # Check if article is recent
                    entry_time = self._parse_entry_time(entry)
                    if datetime.now() - entry_time > self.article_age_limit:
                        continue
                    
                    # Extract content
                    content = self._extract_content(entry)
                    if len(content) < self.min_content_length:
                        continue
                    
                    # Extract symbols
                    symbols = self._extract_symbols(entry.title + ' ' + content)
                    
                    article = NewsArticle(
                        title=entry.title,
                        content=content,
                        source=source_name,
                        url=entry.link,
                        timestamp=entry_time,
                        symbols=symbols,
                        topics=[]
                    )
                    source_articles.append(article)
                
                logger.info(f"✅ Collected {len(source_articles)} articles from {source_name}")
                return source_articles
                
            except Exception as e:
                logger.error(f"❌ Error collecting from {source_name}: {e}")
                return []
        
        # Collect from all RSS feeds concurrently
        tasks = [fetch_feed(name, url) for name, url in self.rss_feeds.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                articles.extend(result)
        
        return articles
    
    async def _collect_news_api(self) -> List[NewsArticle]:
        """Collect articles from News API"""
        if not self.news_api_key:
            logger.warning("⚠️ News API key not configured")
            return []
        
        articles = []
        logger.info("📡 Collecting from News API...")
        
        try:
            # Search for crypto-related news
            search_query = ' OR '.join(self.crypto_keywords)
            url = f"https://newsapi.org/v2/everything"
            
            params = {
                'q': search_query,
                'apiKey': self.news_api_key,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 50
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for article_data in data.get('articles', []):
                        # Parse timestamp
                        timestamp = datetime.fromisoformat(
                            article_data['publishedAt'].replace('Z', '+00:00')
                        )
                        
                        if datetime.now() - timestamp > self.article_age_limit:
                            continue
                        
                        # Extract content and symbols
                        content = f"{article_data.get('title', '')} {article_data.get('description', '')} {article_data.get('content', '')}"
                        content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
                        symbols = self._extract_symbols(content)
                        
                        article = NewsArticle(
                            title=article_data.get('title', ''),
                            content=content,
                            source=article_data.get('source', {}).get('name', 'Unknown'),
                            url=article_data.get('url', ''),
                            timestamp=timestamp,
                            symbols=symbols,
                            topics=[]
                        )
                        articles.append(article)
                
                logger.info(f"✅ Collected {len(articles)} articles from News API")
                
        except Exception as e:
            logger.error(f"❌ Error collecting from News API: {e}")
        
        return articles
    
    async def _collect_social_media(self) -> List[NewsArticle]:
        """Collect articles from social media (Reddit)"""
        articles = []
        logger.info("📱 Collecting from social media...")
        
        # Reddit collection (simplified - would need proper OAuth setup)
        reddit_posts = await self._collect_reddit_posts()
        articles.extend(reddit_posts)
        
        # Twitter collection (would require Twitter API access)
        # twitter_posts = await self._collect_twitter_posts()
        # articles.extend(twitter_posts)
        
        logger.info(f"✅ Collected {len(articles)} articles from social media")
        return articles
    
    async def _collect_reddit_posts(self) -> List[NewsArticle]:
        """Collect posts from relevant Reddit communities"""
        posts = []
        
        # Reddit communities for crypto
        subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'CryptoMarkets']
        
        for subreddit in subreddits:
            try:
                # This is a simplified version - real implementation would need OAuth
                url = f"https://www.reddit.com/r/{subreddit}/new.json"
                headers = {'User-Agent': 'ChloeAI/1.0'}
                
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for post in data.get('data', {}).get('children', [])[:10]:
                            post_data = post.get('data', {})
                            timestamp = datetime.fromtimestamp(post_data.get('created_utc', 0))
                            
                            if datetime.now() - timestamp > self.article_age_limit:
                                continue
                            
                            content = f"{post_data.get('title', '')} {post_data.get('selftext', '')}"
                            symbols = self._extract_symbols(content)
                            
                            if symbols:  # Only include posts mentioning specific symbols
                                article = NewsArticle(
                                    title=post_data.get('title', ''),
                                    content=content[:500],  # Limit content length
                                    source=f"Reddit /r/{subreddit}",
                                    url=f"https://reddit.com{post_data.get('permalink', '')}",
                                    timestamp=timestamp,
                                    symbols=symbols,
                                    topics=['reddit', subreddit]
                                )
                                posts.append(article)
                
            except Exception as e:
                logger.error(f"❌ Error collecting from /r/{subreddit}: {e}")
        
        return posts
    
    def _process_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Process and deduplicate collected articles"""
        # Remove duplicates based on URL
        unique_articles = []
        seen_urls = set()
        
        for article in articles:
            if article.url not in seen_urls:
                # Clean content
                article.content = self._clean_content(article.content)
                
                # Add to unique articles
                unique_articles.append(article)
                seen_urls.add(article.url)
        
        # Sort by timestamp (newest first)
        unique_articles.sort(key=lambda x: x.timestamp, reverse=True)
        
        return unique_articles
    
    def _parse_entry_time(self, entry) -> datetime:
        """Parse timestamp from RSS entry"""
        # Try different timestamp formats
        for time_attr in ['published_parsed', 'updated_parsed']:
            if hasattr(entry, time_attr) and getattr(entry, time_attr):
                return datetime(*getattr(entry, time_attr)[:6])
        
        # Fallback to current time
        return datetime.now()
    
    def _extract_content(self, entry) -> str:
        """Extract content from RSS entry"""
        # Try different content fields
        for content_attr in ['summary', 'content', 'description']:
            if hasattr(entry, content_attr):
                content = getattr(entry, content_attr)
                if isinstance(content, list):
                    content = ' '.join([item.get('value', '') for item in content])
                return str(content)
        
        return ""
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract cryptocurrency symbols from text"""
        text = text.upper()
        found_symbols = []
        
        # Check for common symbols
        for symbol in self.symbols:
            if symbol in text:
                found_symbols.append(symbol)
        
        # Check for common crypto mentions
        crypto_patterns = {
            'BITCOIN': 'BTC',
            'ETHEREUM': 'ETH', 
            'CARDANO': 'ADA',
            'BINANCE': 'BNB',
            'SOLANA': 'SOL',
            'RIPPLE': 'XRP',
            'POLKADOT': 'DOT'
        }
        
        for pattern, symbol in crypto_patterns.items():
            if pattern in text and symbol not in found_symbols:
                found_symbols.append(symbol)
        
        return found_symbols
    
    def _clean_content(self, content: str) -> str:
        """Clean article content"""
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove URLs
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
        
        return content.strip()
    
    def get_articles_by_symbol(self, symbol: str) -> List[NewsArticle]:
        """Get articles mentioning a specific symbol"""
        return [article for article in self.collected_articles 
                if symbol.upper() in [s.upper() for s in article.symbols]]
    
    def get_articles_by_timeframe(self, hours: int = 24) -> List[NewsArticle]:
        """Get recent articles within timeframe"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [article for article in self.collected_articles 
                if article.timestamp >= cutoff_time]
    
    def get_articles_by_source(self, source: str) -> List[NewsArticle]:
        """Get articles from specific source"""
        return [article for article in self.collected_articles 
                if source.lower() in article.source.lower()]

# Example usage
async def main():
    """Example usage of News Collector"""
    async with NewsCollector() as collector:
        articles = await collector.collect_all_news()
        
        print(f"Collected {len(articles)} articles")
        for article in articles[:5]:  # Show first 5
            print(f"\n{article.source}: {article.title}")
            print(f"Symbols: {article.symbols}")
            print(f"Time: {article.timestamp}")
            print(f"Preview: {article.content[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())