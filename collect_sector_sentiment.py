"""
NSE Sector Sentiment Collector
================================
Collects news from free sources (RSS feeds, GDELT, NewsAPI) and computes
sector-level sentiment scores using VADER + FinBERT.

Sentiment is tracked at the SECTOR level (not per-stock) to capture broad
trends like "IT sector headwinds" or "pharma rally on drug approvals"
that the ML models currently miss when generating Buy/Sell signals.

Sources (all free):
  1. RSS Feeds: Economic Times, Moneycontrol, Business Standard, Livemint
  2. GDELT API: Global news tone (unlimited, free)
  3. NewsAPI: Structured search by sector keywords (100 req/day free tier)

NLP Pipeline: VADER (fast, rule-based) + FinBERT (transformer, finance-domain)

Usage:
    python collect_sector_sentiment.py                  # Today's sentiment
    python collect_sector_sentiment.py --backfill 30    # Last 30 days
    python collect_sector_sentiment.py --date 2026-04-03
    python collect_sector_sentiment.py --no-finbert     # VADER only (faster)
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

# NSE 500 sector definitions with Indian market-specific keywords
SECTOR_CONFIG = {
    'Financial Services': {
        'keywords': [
            'HDFC Bank', 'ICICI Bank', 'SBI', 'State Bank of India', 'Axis Bank',
            'Kotak Mahindra', 'Bajaj Finance', 'HDFC Life', 'SBI Life', 'HDFC AMC',
            'Indian banking sector', 'NSE financial stocks', 'RBI rate', 'bank credit',
            'NBFC', 'insurance stocks India', 'fintech India', 'banking stocks NSE',
            'NPA banks India', 'credit growth India',
        ],
        'rss_sectors': ['finance', 'banking'],
        'etf': 'NIFTY_FIN_SERVICE',
    },
    'Information Technology': {
        'keywords': [
            'Infosys', 'TCS', 'Wipro', 'HCL Technologies', 'Tech Mahindra',
            'LTIMindtree', 'Mphasis', 'Persistent Systems', 'Coforge', 'KPIT',
            'IT sector India', 'NIFTY IT', 'Indian software', 'outsourcing India',
            'IT stocks NSE', 'digital transformation India', 'tech hiring India',
            'US visa H1B India', 'IT exports India',
        ],
        'rss_sectors': ['technology', 'tech', 'IT'],
        'etf': 'NIFTY_IT',
    },
    'Healthcare': {
        'keywords': [
            'Apollo Hospitals', 'Max Healthcare', 'Fortis Healthcare', 'Narayana',
            'Aster DM', 'healthcare stocks India', 'hospital stocks NSE',
            'Indian healthcare sector', 'health insurance India', 'medical tourism India',
            'NABH', 'AYUSH India', 'telemedicine India',
        ],
        'rss_sectors': ['healthcare', 'hospitals'],
        'etf': 'NIFTY_HEALTH',
    },
    'Pharmaceuticals': {
        'keywords': [
            'Sun Pharma', 'Dr Reddy', 'Cipla', 'Lupin', 'Divi Laboratories',
            'Biocon', 'Torrent Pharma', 'Zydus', 'Abbott India', 'Alkem',
            'pharma stocks India', 'USFDA India', 'drug approval India', 'DCGI',
            'generic drugs India', 'API manufacturers', 'NIFTY Pharma', 'biosimilar India',
            'pharmaceutical sector NSE',
        ],
        'rss_sectors': ['pharma', 'biotech', 'healthcare'],
        'etf': 'NIFTY_PHARMA',
    },
    'Automobile': {
        'keywords': [
            'Maruti Suzuki', 'Tata Motors', 'Mahindra', 'Bajaj Auto', 'Hero MotoCorp',
            'TVS Motor', 'Eicher Motors', 'MRF', 'Bosch India', 'Motherson Sumi',
            'EV India', 'electric vehicle India', 'auto sales India', 'SIAM India',
            'NIFTY Auto', 'auto stocks NSE', 'India automobile sector',
            'passenger vehicles India', 'two-wheeler India',
        ],
        'rss_sectors': ['auto', 'automobile'],
        'etf': 'NIFTY_AUTO',
    },
    'Consumer Goods': {
        'keywords': [
            'HUL', 'Hindustan Unilever', 'ITC', 'Nestle India', 'Titan Company',
            'Dabur', 'Marico', 'Godrej Consumer', 'Emami', 'Britannia',
            'FMCG stocks India', 'NIFTY FMCG', 'consumer staples India',
            'rural demand India', 'FMCG volume growth', 'consumer stocks NSE',
            'premiumization India', 'quick commerce India',
        ],
        'rss_sectors': ['consumer', 'FMCG'],
        'etf': 'NIFTY_FMCG',
    },
    'Energy': {
        'keywords': [
            'Reliance Industries', 'ONGC', 'BPCL', 'Indian Oil', 'HPCL',
            'Adani Green', 'Tata Power', 'NTPC', 'Power Grid', 'Coal India',
            'oil prices India', 'energy stocks NSE', 'NIFTY Energy', 'crude oil India',
            'renewable energy India', 'solar power India', 'gas prices India',
            'Adani Enterprises energy', 'power sector India',
        ],
        'rss_sectors': ['energy', 'oil', 'power'],
        'etf': 'NIFTY_ENERGY',
    },
    'Metals & Mining': {
        'keywords': [
            'Tata Steel', 'JSW Steel', 'Hindalco', 'Vedanta', 'Coal India',
            'NMDC', 'SAIL', 'Hindustan Zinc', 'National Aluminium', 'Jindal Steel',
            'steel prices India', 'metals stocks NSE', 'NIFTY Metal', 'iron ore India',
            'copper prices', 'aluminium prices', 'China steel demand',
            'mining stocks India', 'commodity stocks India',
        ],
        'rss_sectors': ['metals', 'mining', 'commodities'],
        'etf': 'NIFTY_METAL',
    },
    'Real Estate': {
        'keywords': [
            'DLF', 'Godrej Properties', 'Prestige Estates', 'Macrotech', 'Oberoi Realty',
            'Sobha', 'Brigade Enterprises', 'Phoenix Mills', 'Brookfield India',
            'real estate India', 'NIFTY Realty', 'property stocks NSE',
            'housing sales India', 'affordable housing India', 'RERA', 'home loans India',
            'real estate prices India', 'commercial real estate India',
        ],
        'rss_sectors': ['realty', 'real estate'],
        'etf': 'NIFTY_REALTY',
    },
    'Telecom': {
        'keywords': [
            'Bharti Airtel', 'Vodafone Idea', 'Jio', 'Reliance Jio', 'Indus Towers',
            'BSNL', 'Vi India', 'telecom stocks NSE', 'ARPU India', '5G India',
            'spectrum auction India', 'telecom sector India', 'AGR dues India',
            'mobile subscribers India', 'data revenue India',
        ],
        'rss_sectors': ['telecom', 'communication'],
        'etf': 'NIFTY_TELECOM',
    },
    'Infrastructure & Industrials': {
        'keywords': [
            'Larsen Toubro', 'L&T', 'BHEL', 'Siemens India', 'ABB India',
            'Thermax', 'Cummins India', 'Voltas', 'Bharat Electronics', 'HAL',
            'infrastructure India', 'capex India', 'NIFTY Infra', 'industrial stocks NSE',
            'government spending India', 'PLI scheme', 'Make in India',
            'defence stocks India', 'railways India', 'roads India',
        ],
        'rss_sectors': ['infrastructure', 'industrials', 'manufacturing'],
        'etf': 'NIFTY_INFRA',
    },
    'Chemicals': {
        'keywords': [
            'Asian Paints', 'Pidilite', 'Berger Paints', 'SRF', 'Navin Fluorine',
            'Aarti Industries', 'Deepak Nitrite', 'Tata Chemicals', 'Kansai Nerolac',
            'chemical stocks India', 'specialty chemicals India', 'paint stocks India',
            'agrochemicals India', 'China plus one chemicals', 'chemical sector NSE',
        ],
        'rss_sectors': ['chemicals', 'materials'],
        'etf': 'NIFTY_CHEMICALS',
    },
}

# Indian market-level keywords (applies to all sectors)
MARKET_KEYWORDS = [
    'Indian stock market', 'NSE', 'BSE', 'Sensex', 'NIFTY 50', 'Nifty',
    'RBI', 'Reserve Bank India', 'repo rate', 'inflation India', 'CPI India',
    'FII', 'FPI India', 'DII India', 'foreign investment India',
    'India GDP', 'India economy', 'Union Budget India', 'SEBI',
    'rupee dollar', 'INR exchange rate', 'India growth',
    'India market rally', 'Dalal Street', 'stock market crash India',
    'bull market India', 'bear market India', 'India trade deficit',
    'current account deficit India', 'India imports exports',
]

# RSS Feed URLs for Indian financial news
RSS_FEEDS = {
    'economic_times': {
        'urls': [
            'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
        ],
        'name': 'Economic Times',
    },
    'moneycontrol': {
        'urls': [
            'https://www.moneycontrol.com/rss/latestnews.xml',
            'https://www.moneycontrol.com/rss/marketreports.xml',
        ],
        'name': 'Moneycontrol',
    },
    'business_standard': {
        'urls': [
            'https://www.business-standard.com/rss/markets-106.rss',
            'https://www.business-standard.com/rss/home_page_top_stories.rss',
        ],
        'name': 'Business Standard',
    },
    'livemint': {
        'urls': [
            'https://www.livemint.com/rss/markets',
        ],
        'name': 'Livemint',
    },
    'hindu_businessline': {
        'urls': [
            'https://www.thehindubusinessline.com/markets/feeder/default.rss',
        ],
        'name': 'Hindu BusinessLine',
    },
    'reuters_india': {
        'urls': [
            'https://feeds.reuters.com/reuters/INbusinessNews',
        ],
        'name': 'Reuters India',
    },
}


# ============================================================
# Sentiment Analyzer (VADER + FinBERT)
# ============================================================

class SentimentAnalyzer:
    """Dual-model sentiment scoring with VADER (speed) + FinBERT (accuracy)."""

    def __init__(self, use_finbert: bool = True):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

        self.finbert_pipeline = None
        if use_finbert:
            try:
                from transformers import pipeline
                logger.info("Loading FinBERT model (first run downloads ~400MB)...")
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=-1,
                    truncation=True,
                    max_length=512,
                )
                logger.info("FinBERT loaded successfully")
            except Exception as e:
                logger.warning(f"FinBERT not available, using VADER only: {e}")

    def score(self, text: str) -> Dict[str, float]:
        if not text or len(text.strip()) < 10:
            return {'vader_compound': 0, 'finbert_score': 0, 'combined_score': 0, 'label': 'neutral'}

        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']

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

        if self.finbert_pipeline:
            combined = 0.4 * vader_compound + 0.6 * finbert_score
        else:
            combined = vader_compound

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
        return [self.score(t) for t in texts]


# ============================================================
# News Collectors
# ============================================================

class RSSCollector:
    """Collect news headlines from Indian financial RSS feeds."""

    def __init__(self):
        self.feeds = RSS_FEEDS

    def collect(self, target_date: Optional[datetime] = None) -> List[Dict]:
        if target_date is None:
            target_date = datetime.now()

        articles = []
        for source_key, source_info in self.feeds.items():
            for url in source_info['urls']:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries:
                        pub_date = self._parse_date(entry)
                        # Accept today's articles or last 2 days if target is today
                        if pub_date:
                            days_diff = (target_date.date() - pub_date.date()).days
                            if 0 <= days_diff <= 1:
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
        import email.utils
        date_str = entry.get('published', entry.get('updated', ''))
        if not date_str:
            return None
        try:
            parsed = email.utils.parsedate_to_datetime(date_str)
            return parsed.replace(tzinfo=None)
        except Exception:
            try:
                for fmt in ['%a, %d %b %Y %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        return datetime.strptime(date_str[:len(fmt) + 5], fmt)
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
        articles = []

        # Indian context: combine sector keywords with India/NSE
        query_terms = keywords[:4]
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
                        'tone': art.get('tone', 0),
                    })
        except Exception as e:
            logger.debug(f"GDELT error for {sector}: {e}")

        logger.info(f"GDELT collected: {len(articles)} articles for {sector}")
        return articles

    def collect_market(self, target_date: datetime) -> List[Dict]:
        return self.collect_sector('Market', MARKET_KEYWORDS, target_date)


class NewsAPICollector:
    """Collect news from NewsAPI (100 req/day free tier)."""

    NEWSAPI_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str = ''):
        self.api_key = api_key or NEWSAPI_KEY
        self.enabled = bool(self.api_key)
        self._requests_today = 0
        self._max_requests = 90

    def collect_sector(self, sector: str, keywords: List[str],
                       target_date: datetime) -> List[Dict]:
        if not self.enabled or self._requests_today >= self._max_requests:
            return []

        articles = []
        # India-specific query
        query = ' OR '.join(keywords[:3]) + ' India stocks'

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
    text = (article.get('title', '') + ' ' + article.get('summary', '')).lower()
    matched_sectors = []

    for sector, config in sector_config.items():
        keywords = config['keywords']
        matches = sum(1 for kw in keywords if kw.lower() in text)
        if matches >= 1:
            matched_sectors.append(sector)

    return matched_sectors


# ============================================================
# Main Collector Pipeline
# ============================================================

class SectorSentimentCollector:
    """End-to-end pipeline: collect news -> score -> aggregate -> write to DB."""

    def __init__(self, use_finbert: bool = True):
        logger.info("Initializing NSE Sector Sentiment Collector...")

        self.db = SQLServerConnection()
        self.analyzer = SentimentAnalyzer(use_finbert=use_finbert)
        self.rss = RSSCollector()
        self.gdelt = GDELTCollector()
        self.newsapi = NewsAPICollector()

        logger.info("NSE sentiment collector initialized")

    def collect_and_score(self, target_date: Optional[datetime] = None) -> pd.DataFrame:
        if target_date is None:
            target_date = datetime.now()

        date_str = target_date.strftime('%Y-%m-%d')
        print(f"\n[SENTIMENT] Collecting NSE sector sentiment for {date_str}...")

        all_articles = []

        # RSS feeds (general — classify by sector later)
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

            time.sleep(0.5)

        # Market-level articles
        market_articles = self.gdelt.collect_market(target_date)
        for art in market_articles:
            art['_is_market'] = True
        all_articles.extend(market_articles)

        print(f"[SENTIMENT] Total articles collected: {len(all_articles)}")

        if not all_articles:
            print("[SENTIMENT] No articles found -- creating neutral sentiment records")
            return self._create_neutral_records(target_date)

        all_articles = self._deduplicate(all_articles)
        print(f"[SENTIMENT] After dedup: {len(all_articles)} unique articles")

        print("[SENTIMENT] Scoring sentiment (VADER + FinBERT)...")
        for article in all_articles:
            text = article['title']
            if article.get('summary'):
                text += '. ' + article['summary'][:200]
            article['sentiment'] = self.analyzer.score(text)

        sector_articles: Dict[str, List[Dict]] = {s: [] for s in SECTOR_CONFIG}
        sector_articles['_MARKET'] = []

        for article in all_articles:
            if article.get('_is_market'):
                sector_articles['_MARKET'].append(article)
                continue

            if '_sector_hint' in article:
                sector_articles[article['_sector_hint']].append(article)
                continue

            matched = classify_article_sectors(article, SECTOR_CONFIG)
            if matched:
                for sector in matched:
                    sector_articles[sector].append(article)
            else:
                sector_articles['_MARKET'].append(article)

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
                'confidence': min(agg['news_count'] / 10.0, 1.0),
                'sentiment_momentum_3d': 0,
                'sentiment_momentum_7d': 0,
                'sentiment_vs_avg_30d': 0,
                'market_sentiment_score': market_sentiment['combined_score'],
                'market_news_count': market_sentiment['news_count'],
                'sources': agg['sources'][:500],
            })

        df = pd.DataFrame(records)

        for _, row in df.iterrows():
            indicator = '[+]' if row['sentiment_score'] > 0.1 else ('[-]' if row['sentiment_score'] < -0.1 else '[=]')
            print(f"  {indicator} {row['sector']:35s} score={row['sentiment_score']:+.3f}  "
                  f"articles={row['news_count']:3d}  confidence={row['confidence']:.1f}")

        return df

    def _aggregate_sentiment(self, articles: List[Dict]) -> Dict:
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
        seen_titles = set()
        unique = []
        for art in articles:
            title_key = re.sub(r'[^a-z0-9]', '', art.get('title', '').lower())[:80]
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                unique.append(art)
        return unique

    def _create_neutral_records(self, target_date: datetime) -> pd.DataFrame:
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
        """Update 3d/7d momentum and 30-day z-score features in DB."""
        print("[SENTIMENT] Computing momentum features...")

        try:
            engine = self.db.get_sqlalchemy_engine()
            date_str = target_date.strftime('%Y-%m-%d')

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
            WHERE curr.trading_date = '{date_str}'
            """

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
            WHERE curr.trading_date = '{date_str}'
            """

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
            WHERE curr.trading_date = '{date_str}'
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

        print(f"[SENTIMENT] Saving {len(df)} NSE sector sentiment records to DB...")

        try:
            engine = self.db.get_sqlalchemy_engine()
            date_str = target_date.strftime('%Y-%m-%d')

            from sqlalchemy import text as sql_text
            with engine.begin() as conn:
                conn.execute(sql_text(
                    f"DELETE FROM dbo.nse_sector_sentiment WHERE trading_date = '{date_str}'"
                ))

            df_to_save = df.drop(columns=['sources'], errors='ignore').copy()
            df_to_save['last_updated'] = datetime.now()

            df_to_save.to_sql('nse_sector_sentiment', engine, schema='dbo',
                              if_exists='append', index=False, method='multi', chunksize=100)

            print(f"[SENTIMENT] Saved {len(df_to_save)} records for {date_str}")

        except Exception as e:
            logger.error(f"Failed to save sentiment to DB: {e}")
            raise

    def run(self, target_date: Optional[datetime] = None):
        """Full pipeline: collect -> score -> save -> compute momentum."""
        if target_date is None:
            target_date = datetime.now()

        ensure_sentiment_table(self.db)
        df = self.collect_and_score(target_date)
        self.save_to_db(df, target_date)
        self.compute_momentum_features(target_date)
        print(f"\n[SENTIMENT] NSE sector sentiment pipeline complete for {target_date.strftime('%Y-%m-%d')}")
        return df


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
        print("[SENTIMENT] Table nse_sector_sentiment created successfully")
    else:
        print("[SENTIMENT] Table nse_sector_sentiment already exists")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description='NSE Sector Sentiment Collector')
    parser.add_argument('--date', type=str, default=None,
                        help='Target date (YYYY-MM-DD). Default: today.')
    parser.add_argument('--backfill', type=int, default=None,
                        help='Backfill N days of history.')
    parser.add_argument('--no-finbert', action='store_true',
                        help='Use VADER only (faster, less accurate).')
    args = parser.parse_args()

    use_finbert = not args.no_finbert
    collector = SectorSentimentCollector(use_finbert=use_finbert)

    if args.backfill:
        # Backfill N days
        print(f"\n[SENTIMENT] Backfilling {args.backfill} days of NSE sector sentiment...")
        end_date = datetime.now()
        if args.date:
            end_date = datetime.strptime(args.date, '%Y-%m-%d')

        dates = [end_date - timedelta(days=i) for i in range(args.backfill - 1, -1, -1)]

        success_count = 0
        for dt in dates:
            # Skip weekends (NSE is closed)
            if dt.weekday() >= 5:
                print(f"  [SKIP] {dt.strftime('%Y-%m-%d')} — weekend")
                continue
            try:
                print(f"\n{'='*60}")
                print(f"Processing {dt.strftime('%Y-%m-%d')} ({dt.strftime('%A')})")
                print(f"{'='*60}")
                collector.run(dt)
                success_count += 1
                time.sleep(2)  # Polite pause between days
            except Exception as e:
                print(f"  [ERROR] Failed for {dt.strftime('%Y-%m-%d')}: {e}")
                logger.exception(f"Backfill error for {dt.strftime('%Y-%m-%d')}")

        print(f"\n[SENTIMENT] Backfill complete: {success_count}/{len(dates)} days processed")

    else:
        # Single date
        target_date = datetime.now()
        if args.date:
            target_date = datetime.strptime(args.date, '%Y-%m-%d')
        collector.run(target_date)


if __name__ == '__main__':
    main()
