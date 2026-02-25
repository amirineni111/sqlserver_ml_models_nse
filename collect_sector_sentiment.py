"""
NSE Sector Sentiment Collector
===============================
Collects news from free sources (RSS feeds, GDELT, NewsAPI) and computes
sector-level sentiment scores using VADER + FinBERT.

Sentiment is tracked at the SECTOR level (not per-stock) to capture broad
trends like "IT sector downturn due to AI disruption" that the ML models
currently miss when recommending Buy on dragging software stocks.

Sources (all free):
  1. RSS Feeds: MoneyControl, Economic Times, LiveMint, NDTV Business
  2. GDELT API: Global news tone for India-related coverage (unlimited, free)
  3. NewsAPI: Structured search by sector keywords (100 req/day free tier)

NLP Pipeline: VADER (fast, rule-based) + FinBERT (transformer, finance-domain)

Usage:
    python collect_sector_sentiment.py                  # Today's sentiment
    python collect_sector_sentiment.py --backfill 30    # Last 30 days
    python collect_sector_sentiment.py --date 2026-02-20
"""

import os
import sys
import re
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import feedparser
import requests
from dotenv import load_dotenv

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from database.connection import SQLServerConnection

load_dotenv()

# ============================================================
# Configuration
# ============================================================

logger = logging.getLogger(__name__)

# NewsAPI key (optional — free tier gives 100 req/day)
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')

# Sector definitions with search keywords for news matching
SECTOR_CONFIG = {
    'Information Technology': {
        'keywords': ['IT sector', 'software', 'Infosys', 'TCS', 'Wipro', 'HCL Tech',
                     'Tech Mahindra', 'LTIMindtree', 'Mphasis', 'Coforge', 'Persistent',
                     'Indian IT', 'NIFTY IT', 'technology stocks India',
                     'AI disruption IT', 'outsourcing', 'IT services'],
        'rss_sectors': ['technology', 'it'],
        'nifty_index': 'NIFTY IT',
    },
    'Financial Services': {
        'keywords': ['banking sector', 'HDFC Bank', 'ICICI Bank', 'SBI', 'Kotak',
                     'Bajaj Finance', 'Axis Bank', 'Bank Nifty', 'NIFTY Bank',
                     'Indian banks', 'RBI policy', 'interest rate India',
                     'NPA', 'credit growth', 'financial services India'],
        'rss_sectors': ['banking', 'finance'],
        'nifty_index': 'NIFTY Bank',
    },
    'Healthcare': {
        'keywords': ['pharma sector', 'Sun Pharma', 'Dr Reddy', 'Cipla', 'Divis Labs',
                     'Lupin', 'Aurobindo', 'NIFTY Pharma', 'Indian pharma',
                     'drug approvals India', 'healthcare India', 'biocon',
                     'API exports', 'pharmaceutical stocks'],
        'rss_sectors': ['pharma', 'healthcare'],
        'nifty_index': 'NIFTY Pharma',
    },
    'Automobile': {
        'keywords': ['auto sector', 'Maruti', 'Tata Motors', 'M&M', 'Mahindra',
                     'Bajaj Auto', 'Hero MotoCorp', 'Eicher', 'NIFTY Auto',
                     'auto sales India', 'EV India', 'automobile India',
                     'vehicle sales', 'auto stocks India'],
        'rss_sectors': ['automobile', 'auto'],
        'nifty_index': 'NIFTY Auto',
    },
    'Fast Moving Consumer Goods': {
        'keywords': ['FMCG sector', 'HUL', 'ITC', 'Nestle India', 'Britannia',
                     'Dabur', 'Godrej Consumer', 'Marico', 'NIFTY FMCG',
                     'consumer goods India', 'FMCG stocks', 'rural demand'],
        'rss_sectors': ['fmcg', 'consumer'],
        'nifty_index': 'NIFTY FMCG',
    },
    'Oil & Gas': {
        'keywords': ['oil gas sector', 'Reliance Industries', 'ONGC', 'BPCL', 'IOC',
                     'GAIL', 'Petronet', 'crude oil India', 'natural gas India',
                     'energy stocks India', 'refinery', 'oil prices impact India'],
        'rss_sectors': ['energy', 'oil'],
        'nifty_index': 'NIFTY Energy',
    },
    'Metals & Mining': {
        'keywords': ['metal sector', 'Tata Steel', 'JSW Steel', 'Hindalco', 'Vedanta',
                     'Coal India', 'NMDC', 'NIFTY Metal', 'steel prices India',
                     'mining stocks India', 'commodity prices', 'aluminium India'],
        'rss_sectors': ['metals', 'mining'],
        'nifty_index': 'NIFTY Metal',
    },
    'Realty': {
        'keywords': ['real estate sector', 'DLF', 'Godrej Properties', 'Prestige',
                     'Oberoi Realty', 'Brigade', 'NIFTY Realty', 'housing India',
                     'real estate stocks India', 'property market India'],
        'rss_sectors': ['realty', 'real estate'],
        'nifty_index': 'NIFTY Realty',
    },
}

# Market-level keywords (applies to all sectors)
MARKET_KEYWORDS = [
    'Indian stock market', 'Sensex', 'NIFTY 50', 'BSE', 'NSE India',
    'FII flows', 'DII flows', 'India GDP', 'RBI', 'rupee dollar',
    'Indian economy', 'market crash India', 'bull market India',
    'FPI outflow', 'India inflation', 'India trade deficit',
]

# RSS Feed URLs for Indian financial news
RSS_FEEDS = {
    'moneycontrol': {
        'urls': [
            'https://www.moneycontrol.com/rss/marketreports.xml',
            'https://www.moneycontrol.com/rss/business.xml',
            'https://www.moneycontrol.com/rss/stocksinnews.xml',
        ],
        'name': 'MoneyControl',
    },
    'economictimes': {
        'urls': [
            'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
        ],
        'name': 'Economic Times',
    },
    'livemint': {
        'urls': [
            'https://www.livemint.com/rss/markets',
            'https://www.livemint.com/rss/companies',
        ],
        'name': 'LiveMint',
    },
    'ndtv': {
        'urls': [
            'https://feeds.feedburner.com/ndtvprofit-latest',
        ],
        'name': 'NDTV Profit',
    },
}


# ============================================================
# Sentiment Analyzer (VADER + FinBERT)
# ============================================================

class SentimentAnalyzer:
    """Dual-model sentiment scoring with VADER (speed) + FinBERT (accuracy)."""
    
    def __init__(self, use_finbert: bool = True):
        """Initialize sentiment models.
        
        Args:
            use_finbert: If True, load FinBERT for financial text. Falls back to VADER-only.
        """
        # VADER — always available, fast
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # FinBERT — optional, higher quality for financial text
        self.finbert_pipeline = None
        if use_finbert:
            try:
                from transformers import pipeline
                logger.info("Loading FinBERT model (first run downloads ~400MB)...")
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=-1,  # CPU (use 0 for GPU)
                    truncation=True,
                    max_length=512,
                )
                logger.info("FinBERT loaded successfully")
            except Exception as e:
                logger.warning(f"FinBERT not available, using VADER only: {e}")
    
    def score(self, text: str) -> Dict[str, float]:
        """Score a single text string with both models.
        
        Returns:
            dict with vader_compound, finbert_score, combined_score, label
        """
        if not text or len(text.strip()) < 10:
            return {'vader_compound': 0, 'finbert_score': 0, 'combined_score': 0, 'label': 'neutral'}
        
        # VADER scoring
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']  # -1 to +1
        
        # FinBERT scoring
        finbert_score = 0.0
        if self.finbert_pipeline:
            try:
                result = self.finbert_pipeline(text[:512])[0]
                label = result['label'].lower()
                confidence = result['score']
                if label == 'positive':
                    finbert_score = confidence
                elif label == 'negative':
                    finbert_score = -confidence
                else:
                    finbert_score = 0.0
            except Exception:
                finbert_score = 0.0
        
        # Combined: 40% VADER + 60% FinBERT (FinBERT is better for financial text)
        if self.finbert_pipeline:
            combined = 0.4 * vader_compound + 0.6 * finbert_score
        else:
            combined = vader_compound
        
        # Label
        if combined > 0.15:
            label = 'positive'
        elif combined < -0.15:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'vader_compound': round(vader_compound, 4),
            'finbert_score': round(finbert_score, 4),
            'combined_score': round(combined, 4),
            'label': label,
        }
    
    def score_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Score multiple texts efficiently."""
        return [self.score(t) for t in texts]


# ============================================================
# News Collectors
# ============================================================

class RSSCollector:
    """Collect news headlines from Indian financial RSS feeds."""
    
    def __init__(self):
        self.feeds = RSS_FEEDS
    
    def collect(self, target_date: Optional[datetime] = None) -> List[Dict]:
        """Fetch all RSS feeds and return articles.
        
        Args:
            target_date: Filter articles to this date. None = today.
        
        Returns:
            List of dicts with keys: title, summary, published, source, url
        """
        if target_date is None:
            target_date = datetime.now()
        
        articles = []
        for source_key, source_info in self.feeds.items():
            for url in source_info['urls']:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries:
                        pub_date = self._parse_date(entry)
                        if pub_date and pub_date.date() == target_date.date():
                            articles.append({
                                'title': entry.get('title', ''),
                                'summary': entry.get('summary', entry.get('description', '')),
                                'published': pub_date,
                                'source': source_info['name'],
                                'url': entry.get('link', ''),
                            })
                except Exception as e:
                    logger.debug(f"RSS feed error ({source_key}): {e}")
        
        logger.info(f"RSS collected: {len(articles)} articles for {target_date.date()}")
        return articles
    
    def _parse_date(self, entry) -> Optional[datetime]:
        """Parse the published date from an RSS entry."""
        import email.utils
        date_str = entry.get('published', entry.get('updated', ''))
        if not date_str:
            return None
        try:
            parsed = email.utils.parsedate_to_datetime(date_str)
            return parsed.replace(tzinfo=None)
        except Exception:
            try:
                # Try common formats
                for fmt in ['%a, %d %b %Y %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        return datetime.strptime(date_str[:len(fmt)+5], fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
        return None


class GDELTCollector:
    """Collect sentiment data from GDELT API (free, unlimited)."""
    
    GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    def collect_sector(self, sector: str, keywords: List[str], 
                       target_date: datetime) -> List[Dict]:
        """Fetch GDELT articles for a specific sector.
        
        Args:
            sector: Sector name
            keywords: Search keywords for this sector
            target_date: Date to search for
        
        Returns:
            List of article dicts
        """
        articles = []
        
        # Build query: combine top keywords with India context
        query_terms = keywords[:5]  # Use top 5 keywords to avoid too-broad queries
        query = f'({" OR ".join(query_terms)}) India'
        
        date_str = target_date.strftime('%Y%m%d%H%M%S')
        date_end = (target_date + timedelta(days=1)).strftime('%Y%m%d%H%M%S')
        
        params = {
            'query': query,
            'mode': 'artlist',
            'maxrecords': 50,
            'format': 'json',
            'startdatetime': date_str,
            'enddatetime': date_end,
            'sourcelang': 'eng',
        }
        
        try:
            resp = requests.get(self.GDELT_DOC_API, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for art in data.get('articles', []):
                    articles.append({
                        'title': art.get('title', ''),
                        'summary': '',
                        'published': target_date,
                        'source': f"GDELT:{art.get('domain', 'unknown')}",
                        'url': art.get('url', ''),
                        'tone': art.get('tone', 0),  # GDELT's own tone score
                    })
        except Exception as e:
            logger.debug(f"GDELT error for {sector}: {e}")
        
        logger.info(f"GDELT collected: {len(articles)} articles for {sector}")
        return articles
    
    def collect_market(self, target_date: datetime) -> List[Dict]:
        """Fetch market-level GDELT articles."""
        return self.collect_sector('Market', MARKET_KEYWORDS, target_date)


class NewsAPICollector:
    """Collect news from NewsAPI (100 req/day free tier)."""
    
    NEWSAPI_URL = "https://newsapi.org/v2/everything"
    
    def __init__(self, api_key: str = ''):
        self.api_key = api_key or NEWSAPI_KEY
        self.enabled = bool(self.api_key)
        self._requests_today = 0
        self._max_requests = 90  # Stay under 100 limit
    
    def collect_sector(self, sector: str, keywords: List[str],
                       target_date: datetime) -> List[Dict]:
        """Fetch NewsAPI articles for a sector."""
        if not self.enabled or self._requests_today >= self._max_requests:
            return []
        
        articles = []
        # Use compact query
        query = ' OR '.join(keywords[:3]) + ' India'
        
        params = {
            'q': query,
            'from': target_date.strftime('%Y-%m-%d'),
            'to': target_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 30,
            'apiKey': self.api_key,
        }
        
        try:
            resp = requests.get(self.NEWSAPI_URL, params=params, timeout=30)
            self._requests_today += 1
            
            if resp.status_code == 200:
                data = resp.json()
                for art in data.get('articles', []):
                    articles.append({
                        'title': art.get('title', ''),
                        'summary': art.get('description', ''),
                        'published': target_date,
                        'source': f"NewsAPI:{art.get('source', {}).get('name', 'unknown')}",
                        'url': art.get('url', ''),
                    })
            elif resp.status_code == 426:
                logger.warning("NewsAPI: Free tier date range exceeded (max 1 month back)")
                self.enabled = False
        except Exception as e:
            logger.debug(f"NewsAPI error for {sector}: {e}")
        
        return articles


# ============================================================
# Sector Matcher
# ============================================================

def classify_article_sectors(article: Dict, sector_config: Dict) -> List[str]:
    """Determine which sectors an article is relevant to.
    
    Args:
        article: Article dict with title, summary
        sector_config: SECTOR_CONFIG mapping
    
    Returns:
        List of matching sector names
    """
    text = (article.get('title', '') + ' ' + article.get('summary', '')).lower()
    matched_sectors = []
    
    for sector, config in sector_config.items():
        keywords = config['keywords']
        # Count keyword matches (case-insensitive)
        matches = sum(1 for kw in keywords if kw.lower() in text)
        if matches >= 1:
            matched_sectors.append(sector)
    
    return matched_sectors


# ============================================================
# Main Collector Pipeline
# ============================================================

class SectorSentimentCollector:
    """End-to-end pipeline: collect news → score → aggregate → write to DB."""
    
    def __init__(self, use_finbert: bool = True):
        """Initialize all collectors and the sentiment analyzer."""
        logger.info("Initializing Sector Sentiment Collector...")
        
        self.db = SQLServerConnection()
        self.analyzer = SentimentAnalyzer(use_finbert=use_finbert)
        self.rss = RSSCollector()
        self.gdelt = GDELTCollector()
        self.newsapi = NewsAPICollector()
        
        logger.info("Sentiment collector initialized")
    
    def collect_and_score(self, target_date: Optional[datetime] = None) -> pd.DataFrame:
        """Collect news for all sectors, score sentiment, and return aggregated DataFrame.
        
        Args:
            target_date: Date to collect for. None = today.
        
        Returns:
            DataFrame with one row per sector, containing sentiment scores.
        """
        if target_date is None:
            target_date = datetime.now()
        
        date_str = target_date.strftime('%Y-%m-%d')
        print(f"\n[SENTIMENT] Collecting sector sentiment for {date_str}...")
        
        # 1. Collect from all sources
        all_articles = []
        
        # RSS feeds (general — we classify by sector later)
        rss_articles = self.rss.collect(target_date)
        all_articles.extend(rss_articles)
        
        # GDELT & NewsAPI per sector
        for sector, config in SECTOR_CONFIG.items():
            gdelt_articles = self.gdelt.collect_sector(sector, config['keywords'], target_date)
            for art in gdelt_articles:
                art['_sector_hint'] = sector
            all_articles.extend(gdelt_articles)
            
            newsapi_articles = self.newsapi.collect_sector(sector, config['keywords'], target_date)
            for art in newsapi_articles:
                art['_sector_hint'] = sector
            all_articles.extend(newsapi_articles)
            
            time.sleep(0.5)  # Rate limit politeness
        
        # Market-level articles
        market_articles = self.gdelt.collect_market(target_date)
        for art in market_articles:
            art['_is_market'] = True
        all_articles.extend(market_articles)
        
        print(f"[SENTIMENT] Total articles collected: {len(all_articles)}")
        
        if not all_articles:
            print("[SENTIMENT] No articles found — creating neutral sentiment records")
            return self._create_neutral_records(target_date)
        
        # 2. Deduplicate by title similarity
        all_articles = self._deduplicate(all_articles)
        print(f"[SENTIMENT] After dedup: {len(all_articles)} unique articles")
        
        # 3. Score all articles
        print("[SENTIMENT] Scoring sentiment (VADER + FinBERT)...")
        for article in all_articles:
            text = article['title']
            if article.get('summary'):
                text += '. ' + article['summary'][:200]
            article['sentiment'] = self.analyzer.score(text)
        
        # 4. Classify articles into sectors
        sector_articles: Dict[str, List[Dict]] = {s: [] for s in SECTOR_CONFIG}
        sector_articles['_MARKET'] = []
        
        for article in all_articles:
            # Market articles
            if article.get('_is_market'):
                sector_articles['_MARKET'].append(article)
                continue
            
            # Sector-hinted articles (from GDELT/NewsAPI targeted search)
            if '_sector_hint' in article:
                sector_articles[article['_sector_hint']].append(article)
                continue
            
            # RSS articles — classify by keyword matching
            matched = classify_article_sectors(article, SECTOR_CONFIG)
            if matched:
                for sector in matched:
                    sector_articles[sector].append(article)
            else:
                # Unmatched = market-level
                sector_articles['_MARKET'].append(article)
        
        # 5. Aggregate per sector
        market_sentiment = self._aggregate_sentiment(sector_articles.get('_MARKET', []))
        
        records = []
        for sector in SECTOR_CONFIG:
            articles = sector_articles.get(sector, [])
            agg = self._aggregate_sentiment(articles)
            
            records.append({
                'trading_date': target_date.date(),
                'sector': sector,
                'sentiment_score': agg['combined_score'],
                'vader_score': agg['vader_score'],
                'finbert_score': agg['finbert_score'],
                'positive_ratio': agg['positive_ratio'],
                'negative_ratio': agg['negative_ratio'],
                'neutral_ratio': agg['neutral_ratio'],
                'news_count': agg['news_count'],
                'source_count': agg['source_count'],
                'confidence': min(agg['news_count'] / 10.0, 1.0),  # 10+ articles = full confidence
                'sentiment_momentum_3d': 0,  # Computed after insert via UPDATE
                'sentiment_momentum_7d': 0,
                'sentiment_vs_avg_30d': 0,
                'market_sentiment_score': market_sentiment['combined_score'],
                'market_news_count': market_sentiment['news_count'],
                'sources': agg['sources'][:500],
            })
        
        df = pd.DataFrame(records)
        
        # Print summary
        for _, row in df.iterrows():
            emoji = '🟢' if row['sentiment_score'] > 0.1 else ('🔴' if row['sentiment_score'] < -0.1 else '⚪')
            print(f"  {emoji} {row['sector']:35s} score={row['sentiment_score']:+.3f}  "
                  f"articles={row['news_count']:3d}  confidence={row['confidence']:.1f}")
        
        return df
    
    def _aggregate_sentiment(self, articles: List[Dict]) -> Dict:
        """Aggregate sentiment scores from multiple articles."""
        if not articles:
            return {
                'combined_score': 0, 'vader_score': 0, 'finbert_score': 0,
                'positive_ratio': 0, 'negative_ratio': 0, 'neutral_ratio': 0,
                'news_count': 0, 'source_count': 0, 'sources': '',
            }
        
        sentiments = [a['sentiment'] for a in articles if 'sentiment' in a]
        if not sentiments:
            return {
                'combined_score': 0, 'vader_score': 0, 'finbert_score': 0,
                'positive_ratio': 0, 'negative_ratio': 0, 'neutral_ratio': 0,
                'news_count': len(articles), 'source_count': 0, 'sources': '',
            }
        
        n = len(sentiments)
        labels = [s['label'] for s in sentiments]
        sources = list(set(a.get('source', '') for a in articles))
        
        return {
            'combined_score': round(np.mean([s['combined_score'] for s in sentiments]), 4),
            'vader_score': round(np.mean([s['vader_compound'] for s in sentiments]), 4),
            'finbert_score': round(np.mean([s['finbert_score'] for s in sentiments]), 4),
            'positive_ratio': round(labels.count('positive') / n, 3),
            'negative_ratio': round(labels.count('negative') / n, 3),
            'neutral_ratio': round(labels.count('neutral') / n, 3),
            'news_count': n,
            'source_count': len(sources),
            'sources': ', '.join(sources[:10]),
        }
    
    def _deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles by title similarity."""
        seen_titles = set()
        unique = []
        for art in articles:
            # Normalize title for dedup
            title_key = re.sub(r'[^a-z0-9]', '', art.get('title', '').lower())[:80]
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                unique.append(art)
        return unique
    
    def _create_neutral_records(self, target_date: datetime) -> pd.DataFrame:
        """Create neutral sentiment records when no articles are found."""
        records = []
        for sector in SECTOR_CONFIG:
            records.append({
                'trading_date': target_date.date(),
                'sector': sector,
                'sentiment_score': 0, 'vader_score': 0, 'finbert_score': 0,
                'positive_ratio': 0, 'negative_ratio': 0, 'neutral_ratio': 0,
                'news_count': 0, 'source_count': 0, 'confidence': 0,
                'sentiment_momentum_3d': 0, 'sentiment_momentum_7d': 0,
                'sentiment_vs_avg_30d': 0,
                'market_sentiment_score': 0, 'market_news_count': 0,
                'sources': '',
            })
        return pd.DataFrame(records)
    
    def compute_momentum_features(self, target_date: datetime):
        """Update momentum/z-score features using historical data in DB.
        
        Must run AFTER the current day's raw scores are inserted.
        """
        print("[SENTIMENT] Computing momentum features...")
        
        try:
            engine = self.db.get_sqlalchemy_engine()
            
            # 3-day momentum: current - 3 days ago
            update_3d = f"""
            UPDATE curr
            SET curr.sentiment_momentum_3d = curr.sentiment_score - ISNULL(prev.sentiment_score, 0)
            FROM dbo.nse_sector_sentiment curr
            LEFT JOIN dbo.nse_sector_sentiment prev
                ON curr.sector = prev.sector
                AND prev.trading_date = (
                    SELECT MAX(trading_date) FROM dbo.nse_sector_sentiment
                    WHERE sector = curr.sector 
                    AND trading_date < curr.trading_date
                    AND trading_date >= DATEADD(day, -5, curr.trading_date)
                )
            WHERE curr.trading_date = '{target_date.strftime('%Y-%m-%d')}'
            """
            
            # 7-day momentum: current - 7 days ago
            update_7d = f"""
            UPDATE curr
            SET curr.sentiment_momentum_7d = curr.sentiment_score - ISNULL(prev.sentiment_score, 0)
            FROM dbo.nse_sector_sentiment curr
            LEFT JOIN dbo.nse_sector_sentiment prev
                ON curr.sector = prev.sector
                AND prev.trading_date = (
                    SELECT MAX(trading_date) FROM dbo.nse_sector_sentiment
                    WHERE sector = curr.sector 
                    AND trading_date < curr.trading_date
                    AND trading_date >= DATEADD(day, -10, curr.trading_date)
                )
            WHERE curr.trading_date = '{target_date.strftime('%Y-%m-%d')}'
            """
            
            # Z-score vs 30-day average
            update_zscore = f"""
            UPDATE curr
            SET curr.sentiment_vs_avg_30d = 
                CASE 
                    WHEN hist.std_score > 0.01 
                    THEN (curr.sentiment_score - hist.avg_score) / hist.std_score
                    ELSE 0 
                END
            FROM dbo.nse_sector_sentiment curr
            CROSS APPLY (
                SELECT AVG(sentiment_score) as avg_score, STDEV(sentiment_score) as std_score
                FROM dbo.nse_sector_sentiment
                WHERE sector = curr.sector
                AND trading_date >= DATEADD(day, -30, curr.trading_date)
                AND trading_date < curr.trading_date
            ) hist
            WHERE curr.trading_date = '{target_date.strftime('%Y-%m-%d')}'
            """
            
            from sqlalchemy import text as sql_text
            with engine.begin() as conn:
                conn.execute(sql_text(update_3d))
                conn.execute(sql_text(update_7d))
                conn.execute(sql_text(update_zscore))
            
            print("[SENTIMENT] Momentum features updated")
            
        except Exception as e:
            logger.warning(f"Could not compute momentum features: {e}")
    
    def save_to_db(self, df: pd.DataFrame, target_date: datetime):
        """Save sentiment scores to SQL Server, upserting by (trading_date, sector)."""
        if df.empty:
            return
        
        print(f"[SENTIMENT] Saving {len(df)} sector sentiment records to DB...")
        
        try:
            engine = self.db.get_sqlalchemy_engine()
            
            # Delete existing records for this date (upsert pattern)
            date_str = target_date.strftime('%Y-%m-%d')
            from sqlalchemy import text as sql_text
            with engine.begin() as conn:
                conn.execute(sql_text(
                    f"DELETE FROM dbo.nse_sector_sentiment WHERE trading_date = '{date_str}'"
                ))
            
            # Insert new records
            df['last_updated'] = datetime.now()
            df.to_sql('nse_sector_sentiment', engine, schema='dbo',
                       if_exists='append', index=False)
            
            print(f"[SENTIMENT] Saved {len(df)} records for {date_str}")
            
            # Now compute momentum features using historical data
            self.compute_momentum_features(target_date)
            
        except Exception as e:
            logger.error(f"Error saving sentiment to DB: {e}")
            raise
    
    def run(self, target_date: Optional[datetime] = None):
        """Full pipeline: collect → score → save → compute momentum."""
        if target_date is None:
            target_date = datetime.now()
        
        df = self.collect_and_score(target_date)
        self.save_to_db(df, target_date)
        
        return df
    
    def backfill(self, days: int = 30):
        """Run sentiment collection for the last N days.
        
        Note: RSS feeds only have recent articles. GDELT/NewsAPI support historical.
        For backfill, GDELT is the primary source.
        """
        print(f"\n[SENTIMENT] Backfilling {days} days of sentiment data...")
        
        for i in range(days, 0, -1):
            target_date = datetime.now() - timedelta(days=i)
            # Skip weekends
            if target_date.weekday() >= 5:
                continue
            try:
                self.run(target_date)
                time.sleep(2)  # Be nice to APIs
            except Exception as e:
                logger.warning(f"Backfill failed for {target_date.date()}: {e}")


# ============================================================
# Ensure DB Table Exists
# ============================================================

def ensure_sentiment_table(db: SQLServerConnection):
    """Create the nse_sector_sentiment table if it doesn't exist."""
    check_query = """
    SELECT COUNT(*) as cnt 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME = 'nse_sector_sentiment' AND TABLE_SCHEMA = 'dbo'
    """
    result = db.execute_query(check_query)
    if result.iloc[0]['cnt'] == 0:
        print("[SENTIMENT] Creating nse_sector_sentiment table...")
        create_sql = """
        CREATE TABLE dbo.nse_sector_sentiment (
            id                      INT IDENTITY(1,1) PRIMARY KEY,
            trading_date            DATE NOT NULL,
            sector                  VARCHAR(100) NOT NULL,
            sentiment_score         FLOAT NOT NULL DEFAULT 0,
            vader_score             FLOAT DEFAULT 0,
            finbert_score           FLOAT DEFAULT 0,
            positive_ratio          FLOAT DEFAULT 0,
            negative_ratio          FLOAT DEFAULT 0,
            neutral_ratio           FLOAT DEFAULT 0,
            news_count              INT DEFAULT 0,
            source_count            INT DEFAULT 0,
            confidence              FLOAT DEFAULT 0,
            sentiment_momentum_3d   FLOAT DEFAULT 0,
            sentiment_momentum_7d   FLOAT DEFAULT 0,
            sentiment_vs_avg_30d    FLOAT DEFAULT 0,
            market_sentiment_score  FLOAT DEFAULT 0,
            market_news_count       INT DEFAULT 0,
            sources                 VARCHAR(500) DEFAULT '',
            last_updated            DATETIME DEFAULT GETDATE(),
            CONSTRAINT UQ_nse_sector_sentiment UNIQUE (trading_date, sector)
        )
        """
        from sqlalchemy import text as sql_text
        engine = db.get_sqlalchemy_engine()
        with engine.begin() as conn:
            conn.execute(sql_text(create_sql))
        print("[SENTIMENT] Table created successfully")
    else:
        print("[SENTIMENT] Table nse_sector_sentiment already exists")


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NSE Sector Sentiment Collector')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--backfill', type=int, help='Backfill N days of sentiment data')
    parser.add_argument('--no-finbert', action='store_true', help='Skip FinBERT, use VADER only')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/sentiment_{datetime.now().strftime("%Y%m%d")}.log'),
        ]
    )
    
    # Ensure table exists
    db = SQLServerConnection()
    ensure_sentiment_table(db)
    
    # Initialize collector
    collector = SectorSentimentCollector(use_finbert=not args.no_finbert)
    
    if args.backfill:
        collector.backfill(days=args.backfill)
    else:
        target_date = None
        if args.date:
            target_date = datetime.strptime(args.date, '%Y-%m-%d')
        collector.run(target_date)
    
    print("\n[SENTIMENT] Done!")
