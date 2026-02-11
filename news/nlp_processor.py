"""
NLP Processor for Chloe AI News Analysis
Performs sentiment analysis and topic extraction on news articles
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import Counter

# For demonstration purposes, we'll create a mock sentiment analyzer
# In production, you'd use libraries like transformers, TextBlob, or VADER

logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    compound_score: float  # -1 to 1 (overall sentiment)
    positive_score: float  # 0 to 1
    negative_score: float  # 0 to 1
    neutral_score: float   # 0 to 1
    sentiment_label: str   # POSITIVE, NEGATIVE, NEUTRAL
    confidence: float      # 0 to 1

@dataclass
class TopicResult:
    """Topic extraction result"""
    topics: List[str]
    keywords: List[str]
    topic_scores: Dict[str, float]

class NLPProcessor:
    """
    Natural Language Processing processor for news sentiment analysis
    """
    
    def __init__(self):
        # Sentiment lexicons (simplified for demonstration)
        self.positive_words = {
            'bullish', 'positive', 'gain', 'profit', 'surge', 'rally', 'soar', 'jump',
            'increase', 'rise', 'boost', 'support', 'confidence', 'optimistic',
            'success', 'breakthrough', 'innovation', 'adoption', 'partnership',
            'upgrade', 'outperform', 'beat', 'exceed', 'strong', 'robust'
        }
        
        self.negative_words = {
            'bearish', 'negative', 'loss', 'plunge', 'crash', 'drop', 'fall',
            'decrease', 'decline', 'slump', 'dump', 'sell-off', 'panic',
            'concern', 'worried', 'pessimistic', 'failure', 'scandal',
            'regulation', 'ban', 'restrict', 'downgrade', 'miss', 'underperform',
            'weak', 'volatile', 'uncertainty', 'risk', 'threat'
        }
        
        self.neutral_words = {
            'report', 'analysis', 'study', 'data', 'statistics', 'research',
            'update', 'news', 'announcement', 'statement', 'comment',
            'discussion', 'review', 'examination', 'assessment'
        }
        
        # Topic keywords
        self.topic_keywords = {
            'regulation': ['regulation', 'regulatory', 'compliance', 'law', 'legal', 'government'],
            'technology': ['technology', 'blockchain', 'innovation', 'upgrade', 'development', 'protocol'],
            'market': ['market', 'price', 'trading', 'volume', 'supply', 'demand'],
            'adoption': ['adoption', 'partnership', 'integration', 'mainstream', 'institutional'],
            'security': ['security', 'hack', 'breach', 'vulnerability', 'attack', 'protection'],
            'macro': ['economy', 'inflation', 'interest', 'fed', 'central bank', 'monetary']
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.8, 'strongly': 1.7,
            'significantly': 1.6, 'substantially': 1.5, 'considerably': 1.4,
            'slightly': 0.5, 'mildly': 0.6, 'somewhat': 0.7
        }
        
        # Negation words
        self.negations = {'not', 'no', 'never', 'neither', 'nor', 'nothing', 'nowhere', 'nobody'}
        
        logger.info("🧠 NLP Processor initialized")
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text using rule-based approach
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with scores and label
        """
        if not text or len(text.strip()) == 0:
            return SentimentResult(0.0, 0.0, 0.0, 1.0, 'NEUTRAL', 0.0)
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        words = processed_text.split()
        
        # Calculate sentiment scores
        positive_score = 0.0
        negative_score = 0.0
        neutral_score = 0.0
        word_count = 0
        
        i = 0
        while i < len(words):
            word = words[i].lower()
            intensity = 1.0
            
            # Check for intensifiers
            if i > 0 and words[i-1].lower() in self.intensifiers:
                intensity = self.intensifiers[words[i-1].lower()]
            
            # Check for negations
            is_negated = i > 0 and words[i-1].lower() in self.negations
            
            # Score the word
            if word in self.positive_words:
                score = 0.3 * intensity
                if is_negated:
                    negative_score += score
                else:
                    positive_score += score
                word_count += 1
            elif word in self.negative_words:
                score = 0.3 * intensity
                if is_negated:
                    positive_score += score
                else:
                    negative_score += score
                word_count += 1
            elif word in self.neutral_words:
                neutral_score += 0.1
                word_count += 1
            
            i += 1
        
        # Normalize scores
        if word_count > 0:
            positive_score = min(1.0, positive_score / word_count)
            negative_score = min(1.0, negative_score / word_count)
            neutral_score = min(1.0, neutral_score / word_count)
        else:
            # If no sentiment words found, default to neutral
            positive_score = 0.0
            negative_score = 0.0
            neutral_score = 1.0
        
        # Calculate compound score
        compound_score = positive_score - negative_score
        
        # Determine sentiment label
        if abs(compound_score) < 0.1:
            sentiment_label = 'NEUTRAL'
            confidence = 1.0 - abs(compound_score)
        elif compound_score > 0:
            sentiment_label = 'POSITIVE'
            confidence = compound_score
        else:
            sentiment_label = 'NEGATIVE'
            confidence = abs(compound_score)
        
        return SentimentResult(
            compound_score=compound_score,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            sentiment_label=sentiment_label,
            confidence=confidence
        )
    
    def extract_topics(self, text: str) -> TopicResult:
        """
        Extract topics from text
        
        Args:
            text: Text to analyze
            
        Returns:
            TopicResult with topics and keywords
        """
        if not text:
            return TopicResult([], [], {})
        
        processed_text = self._preprocess_text(text).lower()
        topic_scores = {}
        
        # Score each topic based on keyword frequency
        for topic, keywords in self.topic_keywords.items():
            score = 0
            found_keywords = []
            
            for keyword in keywords:
                count = processed_text.count(keyword.lower())
                if count > 0:
                    score += count
                    found_keywords.append(keyword)
            
            if score > 0:
                topic_scores[topic] = score / len(keywords)  # Normalize by keyword count
        
        # Get top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        top_topics = [topic for topic, score in sorted_topics[:3]]  # Top 3 topics
        topic_scores_dict = dict(sorted_topics[:5])  # Top 5 with scores
        
        # Extract keywords
        all_keywords = []
        for topic in top_topics:
            keywords = self.topic_keywords.get(topic, [])
            all_keywords.extend(keywords)
        
        # Remove duplicates and limit to top keywords
        unique_keywords = list(dict.fromkeys(all_keywords))[:10]
        
        return TopicResult(
            topics=top_topics,
            keywords=unique_keywords,
            topic_scores=topic_scores_dict
        )
    
    def calculate_market_impact(self, sentiment_result: SentimentResult, 
                              topic_result: TopicResult, 
                              article_relevance: float = 1.0) -> float:
        """
        Calculate market impact score based on sentiment and topics
        
        Args:
            sentiment_result: Sentiment analysis result
            topic_result: Topic extraction result
            article_relevance: Relevance score of article to market (0-1)
            
        Returns:
            Market impact score (-1 to 1)
        """
        # Base impact from sentiment
        base_impact = sentiment_result.compound_score * sentiment_result.confidence
        
        # Topic multipliers
        topic_multiplier = 1.0
        high_impact_topics = ['regulation', 'security', 'macro']
        
        for topic in topic_result.topics:
            if topic in high_impact_topics:
                topic_multiplier *= 1.5  # Increase impact for important topics
        
        # Apply relevance weighting
        final_impact = base_impact * topic_multiplier * article_relevance
        
        # Cap at -1 to 1 range
        return max(-1.0, min(1.0, final_impact))
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for multiple texts"""
        return [self.analyze_sentiment(text) for text in texts]
    
    def get_sentiment_trend(self, sentiment_results: List[SentimentResult]) -> Dict:
        """
        Calculate sentiment trend from multiple results
        
        Args:
            sentiment_results: List of sentiment results
            
        Returns:
            Dictionary with trend analysis
        """
        if not sentiment_results:
            return {
                'overall_sentiment': 'NEUTRAL',
                'average_score': 0.0,
                'volatility': 0.0,
                'confidence': 0.0,
                'sample_size': 0
            }
        
        scores = [result.compound_score for result in sentiment_results]
        confidences = [result.confidence for result in sentiment_results]
        labels = [result.sentiment_label for result in sentiment_results]
        
        # Calculate statistics
        avg_score = np.mean(scores)
        volatility = np.std(scores)
        avg_confidence = np.mean(confidences)
        
        # Determine overall sentiment
        if avg_score > 0.1:
            overall_sentiment = 'POSITIVE'
        elif avg_score < -0.1:
            overall_sentiment = 'NEGATIVE'
        else:
            overall_sentiment = 'NEUTRAL'
        
        # Count sentiment distribution
        label_counts = Counter(labels)
        
        return {
            'overall_sentiment': overall_sentiment,
            'average_score': float(avg_score),
            'volatility': float(volatility),
            'confidence': float(avg_confidence),
            'sample_size': len(sentiment_results),
            'distribution': dict(label_counts),
            'extreme_positive': len([s for s in scores if s > 0.5]),
            'extreme_negative': len([s for s in scores if s < -0.5])
        }

class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analyzer with more sophisticated techniques
    """
    
    def __init__(self):
        self.base_analyzer = NLPProcessor()
        self.context_weights = {
            'fear': 1.2,      # Fear-based news has higher impact
            'greed': 1.1,     # Greed-based news has moderate impact
            'fud': 1.3,       # Fear Uncertainty Doubt has high impact
            'fomo': 1.0       # Fear Of Missing Out has normal impact
        }
    
    def analyze_with_context(self, text: str, source_type: str = 'news') -> SentimentResult:
        """
        Analyze sentiment with context awareness
        
        Args:
            text: Text to analyze
            source_type: Type of source (news, social_media, reddit, etc.)
            
        Returns:
            Enhanced sentiment result
        """
        # Get base sentiment
        base_result = self.base_analyzer.analyze_sentiment(text)
        
        # Apply source-specific adjustments
        if source_type == 'social_media':
            # Social media tends to be more extreme
            base_result.compound_score *= 1.2
            base_result.confidence *= 0.8  # Lower confidence due to noise
        elif source_type == 'reddit':
            # Reddit can be very biased
            if abs(base_result.compound_score) > 0.3:
                base_result.compound_score *= 1.1
                base_result.confidence *= 0.7
        elif source_type == 'news':
            # News is generally more reliable
            base_result.confidence *= 1.1
        
        # Apply context weights
        text_lower = text.lower()
        for context, weight in self.context_weights.items():
            if context in text_lower:
                base_result.compound_score *= weight
                break
        
        # Ensure scores stay in valid range
        base_result.compound_score = max(-1.0, min(1.0, base_result.compound_score))
        base_result.confidence = max(0.0, min(1.0, base_result.confidence))
        
        return base_result

# Example usage
def main():
    """Example usage of NLP Processor"""
    processor = NLPProcessor()
    advanced_analyzer = AdvancedSentimentAnalyzer()
    
    # Test texts
    test_texts = [
        "Bitcoin price surges 10% as institutional adoption increases",
        "Ethereum faces regulatory challenges that could impact future growth",
        "Market analysts remain cautiously optimistic about crypto prospects",
        "New partnership announced between major exchanges and traditional banks",
        "Security concerns raised after recent exchange hack incident"
    ]
    
    print("🧠 NLP Sentiment Analysis Demo")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text}")
        
        # Basic analysis
        sentiment = processor.analyze_sentiment(text)
        topics = processor.extract_topics(text)
        
        print(f"   Sentiment: {sentiment.sentiment_label} (Score: {sentiment.compound_score:.3f}, Confidence: {sentiment.confidence:.3f})")
        print(f"   Topics: {', '.join(topics.topics) if topics.topics else 'None'}")
        
        # Advanced analysis
        advanced_sentiment = advanced_analyzer.analyze_with_context(text, 'news')
        print(f"   Advanced Sentiment: {advanced_sentiment.sentiment_label} (Score: {advanced_sentiment.compound_score:.3f})")
        
        # Market impact
        impact = processor.calculate_market_impact(sentiment, topics)
        print(f"   Market Impact: {impact:.3f}")

if __name__ == "__main__":
    main()