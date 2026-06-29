# Import libraries
import warnings
warnings.filterwarnings('ignore')
import os
import re
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from crewai.tools import tool  # Updated import for custom tools
import yfinance as yf
import anthropic
from IPython.display import display, Markdown

# --- Configuration ---
os.environ["CREWAI_TELEMETRY"] = "False"

# API Keys Configuration
@dataclass
class APIConfig:
    # REQUIRED APIs
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")
    serper_api_key: str = os.getenv("SERPER_API_KEY", "")
    
    # FIXED: Model configuration for LiteLLM
    claude_model: str = "claude-sonnet-4-20250514"
    claude_base_url: str = "https://api.anthropic.com"
    
    # PROFESSIONAL: Enhanced rate limiting settings
    max_input_tokens: int = 150000      
    max_output_tokens: int = 8000       
    rate_limit_delay: float = 15.0              # Conservative delays
    inter_agent_delay: float = 25.0             # Between agents  
    context_processing_delay: float = 10.0      # Context processing
    
    # PROFESSIONAL: Strategic token allocation per agent
    technical_agent_tokens: int = 5000          # Technical analysis depth
    sentiment_agent_tokens: int = 5000          # Comprehensive sentiment
    risk_agent_tokens: int = 5000               # Detailed risk scenarios
    portfolio_manager_tokens: int = 8000        # Maximum for decisions    
    
    def validate(self):
        """Validation logic - unchanged"""
        missing_keys = []
        if not self.claude_api_key or self.claude_api_key.startswith("["):
            missing_keys.append("claude_api_key")
        if not self.serper_api_key or self.serper_api_key.startswith("["):
            missing_keys.append("serper_api_key")
        
        if missing_keys:
            print(f"❌ Missing required API keys: {missing_keys}")
            print("🔑 Get your API keys:")
            print("• Claude API: https://console.anthropic.com")
            print("• Serper: https://serper.dev")
            return False
        return True


api_config = APIConfig()
os.environ["SERPER_API_KEY"] = api_config.serper_api_key

# --- Claude Rate Limiting ---
import time
import random

# --- Enhanced LLM Setup with Claude Sonnet 4 ---
def setup_claude_llms(api_config: APIConfig):
    """PROFESSIONAL: Strategic multi-model LLM setup for zero-failure reliability"""
    try:
        # Test connection first
        test_client = anthropic.Anthropic(
            api_key=api_config.claude_api_key,
            base_url=api_config.claude_base_url
        )

        test_message = test_client.messages.create(
            model=api_config.claude_model,
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Test connection. Respond with 'Claude connected successfully.'"}
            ]
        )

        response_text = ""
        if test_message.content and hasattr(test_message.content[0], "text"):
            response_text = test_message.content[0].text

        if "Claude connected successfully" in response_text:
            print("Professional-grade Claude connection validated")
        else:
            print("Claude responded with unexpected content - proceeding anyway")

        # STRATEGIC LLM ASSIGNMENT FOR 4-AGENT SYSTEM
        llm_configs = {}

        # Technical Analysis: Sonnet 4 (Pattern Recognition Excellence)
        llm_configs['technical'] = LLM(
            model="anthropic/claude-sonnet-4-20250514",
            api_key=api_config.claude_api_key,
            max_tokens=api_config.technical_agent_tokens,
            timeout=120,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )

        # Sentiment Analysis: Sonnet 4 (Nuanced Market Psychology)
        llm_configs['sentiment'] = LLM(
            model="anthropic/claude-sonnet-4-20250514",
            api_key=api_config.claude_api_key,
            max_tokens=api_config.sentiment_agent_tokens,
            timeout=120,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )

        # Risk Management: Haiku 3.5 (High Context Processing)
        llm_configs['risk'] = LLM(
            model="anthropic/claude-3-5-haiku-20241022",
            api_key=api_config.claude_api_key,
            max_tokens=api_config.risk_agent_tokens,
            timeout=120,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )

        # Portfolio Manager: Opus 4 for final decisions
        llm_configs['portfolio'] = LLM(
            model="anthropic/claude-opus-4-20250514",
            api_key=api_config.claude_api_key,
            max_tokens=api_config.portfolio_manager_tokens,
            timeout=180,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )

        # Manager LLM: sonnet for efficient coordination
        manager_llm = LLM(
            model="anthropic/claude-3-7-sonnet-20250219",
            api_key=api_config.claude_api_key,
            max_tokens=4000,
            timeout=120,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )

        print("STRATEGIC MODEL ASSIGNMENT COMPLETE:")
        print(f"Technical: Sonnet 4 ({api_config.technical_agent_tokens} tokens)")
        print(f"Sentiment: Sonnet 4 ({api_config.sentiment_agent_tokens} tokens)")
        print(f"Risk: Haiku 3.5 ({api_config.risk_agent_tokens} tokens)")
        print(f"Portfolio: Opus 4 ({api_config.portfolio_manager_tokens} tokens)")
        print("Manager: Sonnet 3.7 (coordination)")

        return llm_configs, manager_llm

    except Exception as e:
        print(f"Strategic LLM initialization failed: {e}")
        return None, None
        
# Direct Claude client for custom tools
def setup_direct_claude_client(api_config: APIConfig):
    """Setup direct Claude client for custom tools - unchanged"""
    return anthropic.Anthropic(
        api_key=api_config.claude_api_key,
        base_url=api_config.claude_base_url  # This still needs claude_base_url
    )

def check_claude_credits(api_config: APIConfig):
    """Check Claude API credits and usage"""
    try:
        client = setup_direct_claude_client(api_config)
        
        # Test with a minimal request to check account status
        test_response = client.messages.create(
            model=api_config.claude_model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        
        print("✅ Claude API account is active and has credits")
        print(f"📊 Model: {api_config.claude_model}")
        print(f"💰 Billing: Check console.anthropic.com for detailed usage")
        return True
        
    except anthropic.AuthenticationError:
        print("❌ Claude API key is invalid")
        return False
    except anthropic.PermissionDeniedError:
        print("❌ Claude API key doesn't have required permissions")
        return False
    except anthropic.RateLimitError:
        print("⚠️ Claude API rate limit exceeded - wait before retrying")
        return False
    except Exception as e:
        print(f"❌ Claude API error: {e}")
        return False

_, manager_llm = setup_claude_llms(api_config)

# --- Smart Yahoo Cache System ---
class SmartYahooCache:
    """
    Minimalist cache that fetches once, slices many times.
    Uses per-component TTLs so price data refreshes frequently while
    fundamentals stay cached for hours.

    Component TTLs:
    - history: 5 minutes (price-sensitive)
    - news: 15 minutes (headlines matter for sentiment)
    - recommendations: 1 hour (analyst updates trickle in)
    - info: 24 hours (fundamentals barely change)
    """

    # Per-component TTLs in seconds
    TTL_HISTORY = 300          # 5 minutes
    TTL_NEWS = 900             # 15 minutes
    TTL_RECOMMENDATIONS = 3600  # 1 hour
    TTL_INFO = 86400           # 24 hours

    def __init__(self):
        # ticker -> {'history': ..., 'history_ts': ...,
        #            'info': ..., 'info_ts': ...,
        #            'news': ..., 'news_ts': ...,
        #            'recommendations': ..., 'recommendations_ts': ...,
        #            'ticker_obj': ...}
        self.ticker_data = {}
        self.api_calls = 0
        self.cache_hits = 0

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        """Canonicalize ticker symbols for consistent cache keys."""
        return ticker.strip().upper() if ticker else ticker

    def _is_fresh(self, entry: dict, component: str, ttl: float) -> bool:
        """Check whether a specific cached component is still within its TTL."""
        ts_key = f"{component}_ts"
        if component not in entry or ts_key not in entry:
            return False
        return (time.time() - entry[ts_key]) < ttl

    def get_ticker_data(self, ticker: str, force_refresh: bool = False):
        """
        Fetch or return cached data for ticker.
        Returns a dict with the same shape callers expect:
        {'history', 'info', 'news', 'recommendations', 'timestamp', 'ticker_obj'}
        Each component is independently refreshed when stale.
        """
        ticker = self._normalize_ticker(ticker)
        entry = self.ticker_data.get(ticker, {})

        # Determine which components need refresh
        needs_history = force_refresh or not self._is_fresh(entry, 'history', self.TTL_HISTORY)
        needs_info = force_refresh or not self._is_fresh(entry, 'info', self.TTL_INFO)
        needs_news = force_refresh or not self._is_fresh(entry, 'news', self.TTL_NEWS)
        needs_recs = force_refresh or not self._is_fresh(entry, 'recommendations', self.TTL_RECOMMENDATIONS)

        # If everything's fresh, return immediately
        if not (needs_history or needs_info or needs_news or needs_recs):
            self.cache_hits += 1
            print(f"✅ CACHE HIT (all fresh): {ticker} (saved API call #{self.cache_hits})")
            return self._unified_view(entry)

        # Log which components are being refreshed
        stale = [c for c, n in [('history', needs_history), ('info', needs_info),
                                ('news', needs_news), ('recommendations', needs_recs)] if n]
        print(f"🌐 Fetching {ticker} components: {stale}")
        self.api_calls += 1

        # Rate-limit courtesy delay between yfinance calls
        if self.api_calls > 1:
            time.sleep(2)

        try:
            stock = entry.get('ticker_obj') or yf.Ticker(ticker)
            entry['ticker_obj'] = stock
            now = time.time()

            if needs_history:
                # Always fetch 2y so any period slice has the data it needs
                entry['history'] = stock.history(period="2y", interval="1d")
                entry['history_ts'] = now

            if needs_info:
                entry['info'] = stock.info
                entry['info_ts'] = now

            if needs_news:
                entry['news'] = stock.news if hasattr(stock, 'news') else []
                entry['news_ts'] = now

            if needs_recs:
                try:
                    entry['recommendations'] = stock.recommendations
                except (AttributeError, KeyError, ValueError, IndexError) as e:
                    # Narrow catch: realistic yfinance failure modes for missing data
                    print(f"ℹ️ Recommendations unavailable for {ticker}: {e}")
                    entry['recommendations'] = pd.DataFrame()
                entry['recommendations_ts'] = now

            self.ticker_data[ticker] = entry
            return self._unified_view(entry)

        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            # Return stale data if we have any rather than failing outright
            if entry and any(k.endswith('_ts') for k in entry):
                print(f"⚠️ Returning stale data for {ticker}")
                return self._unified_view(entry)
            return None

    def _unified_view(self, entry: dict) -> dict:
        """
        Return the dict shape that callers (the tools) expect.
        Keeps a single 'timestamp' for backward compatibility — uses the oldest
        component timestamp so callers can reason about overall staleness.
        """
        component_timestamps = [
            entry.get('history_ts', 0),
            entry.get('info_ts', 0),
            entry.get('news_ts', 0),
            entry.get('recommendations_ts', 0),
        ]
        valid_timestamps = [t for t in component_timestamps if t > 0]
        oldest = min(valid_timestamps) if valid_timestamps else 0

        return {
            'history': entry.get('history'),
            'info': entry.get('info', {}),
            'news': entry.get('news', []),
            'recommendations': entry.get('recommendations', pd.DataFrame()),
            'timestamp': oldest,
            'ticker_obj': entry.get('ticker_obj'),
        }

    def slice_history(self, hist: pd.DataFrame, period: str) -> pd.DataFrame:
        """
        Slice history DataFrame to requested period using trading-day row counts.
        Matches what yfinance would return for the same period parameter.
        """
        if hist is None or hist.empty:
            return hist

        # Trading-day counts (yfinance's actual conventions)
        period_trading_days = {
            "1d": 1,
            "5d": 5,
            "1mo": 22,
            "3mo": 65,
            "6mo": 126,
            "1y": 252,
            "2y": 504,
            "5y": 1260,
            "max": None,
        }

        rows = period_trading_days.get(period)
        if rows is None or rows >= len(hist):
            return hist.copy()

        return hist.tail(rows).copy()

    def get_stats(self) -> str:
        """Return cache statistics."""
        total_requests = self.api_calls + self.cache_hits
        savings = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return f"""
🎯 Smart Yahoo Cache Statistics:
- API Calls Made: {self.api_calls}
- Cache Hits: {self.cache_hits}
- Total Requests: {total_requests}
- Cache Efficiency: {savings:.1f}%
- Tickers Cached: {list(self.ticker_data.keys())}
"""

# Global cache instance
_smart_cache = SmartYahooCache()

def _get_risk_free_rate() -> float:
    """
    Fetch the current 3-month Treasury bill rate from Yahoo Finance (^IRX).
    Falls back to a recent-history default if the fetch fails.
    The ^IRX quote is already in percent, so divide by 100.
    """
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if not hist.empty:
            latest = hist['Close'].iloc[-1]
            # Sanity check: T-bill yields realistically sit between 0% and 20%
            if 0 <= latest <= 20:
                return float(latest) / 100.0
    except (KeyError, IndexError, ValueError, AttributeError) as e:
        print(f"⚠️ Risk-free rate fetch failed ({e}), using fallback")

    # Fallback: a reasonable mid-cycle default. Better than the old 2%.
    return 0.045

@tool
def enhanced_financial_data_tool(ticker: str, period: str = "1y", interval: str = "1d",
                                include_indicators: bool = True) -> str:
    """
    Enhanced financial data tool with smart caching.
    Fetches comprehensive financial data including OHLCV, technical indicators,
    volume analysis, and volatility metrics for any stock ticker.
    NOW WITH SMART CACHING: Fetches once, reuses intelligently.
    """
    global _smart_cache

    # Normalize ticker for consistent cache keys
    ticker = ticker.strip().upper() if ticker else ticker

    # Get cached data (fetches 2y if needed)
    cached = _smart_cache.get_ticker_data(ticker)

    if cached is None:
        return f"Error: Unable to fetch data for {ticker}"

    # Slice to requested period
    hist = _smart_cache.slice_history(cached['history'], period)

    if hist.empty:
        return f"Error: No data found for {ticker} in period {period}"

    # Map interval to annualization factor (periods per year) for volatility
    interval_periods_per_year = {
        "1m": 252 * 6.5 * 60,
        "5m": 252 * 6.5 * 12,
        "15m": 252 * 6.5 * 4,
        "30m": 252 * 6.5 * 2,
        "60m": 252 * 6.5,
        "1h": 252 * 6.5,
        "1d": 252,
        "5d": 252 / 5,
        "1wk": 52,
        "1mo": 12,
        "3mo": 4,
    }
    periods_per_year = interval_periods_per_year.get(interval, 252)

    # Calculate indicators
    if include_indicators and len(hist) > 20:
        # Moving averages
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean() if len(hist) > 50 else np.nan
        hist['EMA_12'] = hist['Close'].ewm(span=12).mean()
        hist['EMA_26'] = hist['Close'].ewm(span=26).mean()

        # Volatility (annualized for the actual bar interval)
        hist['Returns'] = hist['Close'].pct_change()
        hist['Volatility'] = hist['Returns'].rolling(window=20).std() * np.sqrt(periods_per_year)

        # RSI calculation with proper edge-case handling
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Peg edge cases: all-gain -> 100, all-loss -> 0, flat -> 50
        rsi = rsi.where(~((gain > 0) & (loss == 0)), 100.0)
        rsi = rsi.where(~((gain == 0) & (loss > 0)), 0.0)
        rsi = rsi.where(~((gain == 0) & (loss == 0)), 50.0)
        hist['RSI'] = rsi

        # MACD
        hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
        hist['MACD_Signal'] = hist['MACD'].ewm(span=9).mean()

        # Bollinger Bands
        hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
        bb_std = hist['Close'].rolling(window=20).std()
        hist['BB_Upper'] = hist['BB_Middle'] + (bb_std * 2)
        hist['BB_Lower'] = hist['BB_Middle'] - (bb_std * 2)

    # Analysis summary
    current_price = hist['Close'].iloc[-1]
    price_change = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0
    volume_avg = hist['Volume'].tail(20).mean() if len(hist) > 20 else hist['Volume'].mean()
    latest_volume = hist['Volume'].iloc[-1]
    volume_ratio = latest_volume / volume_avg if volume_avg > 0 else 1.0

    # Period-aware high/low labels
    period_labels = {
        "1d": "1-Day", "5d": "5-Day", "1mo": "1-Month", "3mo": "3-Month",
        "6mo": "6-Month", "1y": "52-Week", "2y": "2-Year", "5y": "5-Year",
        "max": "All-Time"
    }
    range_label = period_labels.get(period, period)
    trading_days = len(hist)

    high_in_period = hist['High'].max()
    low_in_period = hist['Low'].min()
    range_pct = ((current_price - low_in_period) / (high_in_period - low_in_period) * 100) \
        if high_in_period > low_in_period else 50.0

    # Safe indicator access with fallbacks
    rsi_value = f"{hist['RSI'].iloc[-1]:.1f}" if 'RSI' in hist.columns and not pd.isna(hist['RSI'].iloc[-1]) else "N/A"
    macd_value = f"{hist['MACD'].iloc[-1]:.3f}" if 'MACD' in hist.columns and not pd.isna(hist['MACD'].iloc[-1]) else "N/A"

    if 'SMA_20' in hist.columns and not pd.isna(hist['SMA_20'].iloc[-1]):
        sma20_pct = f"{((current_price - hist['SMA_20'].iloc[-1]) / hist['SMA_20'].iloc[-1] * 100):.2f}%"
    else:
        sma20_pct = "N/A"

    if 'SMA_50' in hist.columns and not pd.isna(hist['SMA_50'].iloc[-1]):
        sma50_pct = f"{((current_price - hist['SMA_50'].iloc[-1]) / hist['SMA_50'].iloc[-1] * 100):.2f}%"
    else:
        sma50_pct = "N/A"

    volatility_value = (
        f"{hist['Volatility'].iloc[-1]:.2f}"
        if 'Volatility' in hist.columns and not pd.isna(hist['Volatility'].iloc[-1])
        else "N/A"
    )
    bb_upper = f"${hist['BB_Upper'].iloc[-1]:.2f}" if 'BB_Upper' in hist.columns and not pd.isna(hist['BB_Upper'].iloc[-1]) else "N/A"
    bb_lower = f"${hist['BB_Lower'].iloc[-1]:.2f}" if 'BB_Lower' in hist.columns and not pd.isna(hist['BB_Lower'].iloc[-1]) else "N/A"

    analysis = f"""
COMPREHENSIVE ANALYSIS FOR {ticker}:

PRICE DATA:
- Current Price: ${current_price:.2f}
- Daily Change: {price_change:.2f}%
- {range_label} High: ${high_in_period:.2f}
- {range_label} Low: ${low_in_period:.2f}
- Current vs {range_label} range: {range_pct:.1f}% (based on {trading_days} trading days)

VOLUME ANALYSIS:
- Average Volume (20-day): {volume_avg:,.0f}
- Latest Volume: {latest_volume:,.0f}
- Volume Ratio: {volume_ratio:.2f}x

TECHNICAL INDICATORS:
- RSI (14): {rsi_value}
- MACD: {macd_value}
- Price vs SMA20: {sma20_pct}
- Price vs SMA50: {sma50_pct}
- Volatility (20-period, annualized from {interval} bars): {volatility_value}

SUPPORT/RESISTANCE LEVELS:
- Recent Support: ${hist['Low'].tail(20).min():.2f}
- Recent Resistance: ${hist['High'].tail(20).max():.2f}
- Bollinger Upper: {bb_upper}
- Bollinger Lower: {bb_lower}
            """

    return analysis.strip()

@tool
def market_sentiment_tool(ticker: str, days_back: int = 7) -> str:
    """
    Market sentiment analysis tool with smart caching.
    Analyzes market sentiment for a stock from news, social media mentions,
    and analyst ratings to provide comprehensive sentiment score.
    NOW WITH SMART CACHING: Reuses fetched data intelligently.
    """
    global _smart_cache

    # Normalize ticker for consistent cache keys
    ticker = ticker.strip().upper() if ticker else ticker

    # Get cached data
    cached = _smart_cache.get_ticker_data(ticker)

    if cached is None:
        return f"Error analyzing {ticker}: Unable to fetch data"

    info = cached['info']
    company_name = info.get('longName', ticker)

    # Handle analyst recommendations with narrowed exception
    rec_summary = {"No recent recommendations": 1}
    try:
        recommendations = cached['recommendations']
        if recommendations is not None and not recommendations.empty:
            if 'To Grade' in recommendations.columns:
                latest_rec = recommendations.tail(5)
                rec_summary = latest_rec['To Grade'].value_counts().to_dict()
            elif 'Recommendation' in recommendations.columns:
                latest_rec = recommendations.tail(5)
                rec_summary = latest_rec['Recommendation'].value_counts().to_dict()
            else:
                rec_summary = {"Recent recommendations": len(recommendations.tail(5))}
    except (KeyError, AttributeError, TypeError) as e:
        rec_summary = {"Recommendations unavailable": str(e)}

    # News sentiment with word-boundary matching and per-headline scoring
    sentiment_score = 0
    news_titles = []
    articles_analyzed = 0
    news_fetch_status = "ok"

    sentiment_words_positive = [
        'growth', 'profit', 'profits', 'beat', 'beats', 'strong', 'bullish',
        'upgrade', 'upgraded', 'buy', 'gain', 'gains', 'rise', 'rises',
        'high', 'highs', 'rally', 'surge', 'soar', 'jump'
    ]
    sentiment_words_negative = [
        'loss', 'losses', 'decline', 'declines', 'weak', 'bearish',
        'downgrade', 'downgraded', 'sell', 'concern', 'concerns',
        'fall', 'falls', 'drop', 'drops', 'low', 'lows', 'plunge', 'slump'
    ]

    # Word boundaries prevent "beat" from matching "beat down" or "low" matching "lower"
    pos_pattern = re.compile(r'\b(' + '|'.join(sentiment_words_positive) + r')\b', re.IGNORECASE)
    neg_pattern = re.compile(r'\b(' + '|'.join(sentiment_words_negative) + r')\b', re.IGNORECASE)

    try:
        news = cached['news']
        if news:
            news_titles = [item.get('title', '') for item in news[:10] if item.get('title')]
            articles_analyzed = len(news_titles)

            # Per-headline net scoring: each headline contributes at most +1, -1, or 0
            positive_count = 0
            negative_count = 0
            for title in news_titles:
                pos_hits = len(set(m.group(0).lower() for m in pos_pattern.finditer(title)))
                neg_hits = len(set(m.group(0).lower() for m in neg_pattern.finditer(title)))
                if pos_hits > neg_hits:
                    positive_count += 1
                elif neg_hits > pos_hits:
                    negative_count += 1
                # Ties (mixed-sentiment headlines) contribute nothing

            if positive_count + negative_count > 0:
                sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            news_fetch_status = "no_news"
    except (KeyError, TypeError, AttributeError) as e:
        news_fetch_status = f"fetch_error: {e}"

    # Build display-friendly headline list (separate from the real count)
    if articles_analyzed > 0:
        headline_display = news_titles
    elif news_fetch_status == "no_news":
        headline_display = ["(no recent news available)"]
    else:
        headline_display = [f"(news unavailable: {news_fetch_status})"]

    # Interpret news sentiment
    if sentiment_score > 0.3:
        news_sentiment = "Positive"
    elif sentiment_score < -0.3:
        news_sentiment = "Negative"
    else:
        news_sentiment = "Neutral"

    # Analyze analyst sentiment from recommendations
    analyst_sentiment = "Mixed"
    try:
        buy_keywords = ['buy', 'strong buy', 'outperform', 'overweight', 'positive']
        sell_keywords = ['sell', 'strong sell', 'underperform', 'underweight', 'negative']

        buy_count = sum(count for rec, count in rec_summary.items()
                       if any(keyword in str(rec).lower() for keyword in buy_keywords))
        sell_count = sum(count for rec, count in rec_summary.items()
                        if any(keyword in str(rec).lower() for keyword in sell_keywords))

        if buy_count > sell_count * 1.5:
            analyst_sentiment = "Bullish"
        elif sell_count > buy_count * 1.5:
            analyst_sentiment = "Bearish"
    except Exception as e:
        # Log-and-continue: rec_summary shape varies, so broad catch is justified,
        # but surface the failure rather than swallowing it silently.
        print(f"⚠️ Analyst sentiment scoring failed unexpectedly: "
              f"{type(e).__name__}: {e}")
        analyst_sentiment = "Unknown"

    sentiment_analysis = f"""
MARKET SENTIMENT ANALYSIS FOR {ticker} ({company_name}):

ANALYST RECOMMENDATIONS:
{json.dumps(rec_summary, indent=2)}

NEWS SENTIMENT:
- Sentiment Score: {sentiment_score:.2f} (Range: -1 to +1)
- Articles Analyzed: {articles_analyzed}
- Recent Headlines:
{chr(10).join([f"  • {title[:100]}..." if len(title) > 100 else f"  • {title}" for title in headline_display[:5]])}

SENTIMENT INTERPRETATION:
- News Sentiment: {news_sentiment}
- Analyst Sentiment: {analyst_sentiment}
- Overall Market Sentiment: {'Bullish' if news_sentiment == 'Positive' and analyst_sentiment == 'Bullish' else 'Bearish' if news_sentiment == 'Negative' and analyst_sentiment == 'Bearish' else 'Mixed'}
        """

    return sentiment_analysis

@tool
def economic_data_tool(indicators: str = "basic") -> str:
    """
    Economic data tool for fetching relevant economic indicators.
    Fetches relevant economic indicators like interest rates, inflation,
    GDP growth, and unemployment that might affect stock performance.
    """
    try:
        economic_summary = """
CURRENT ECONOMIC ENVIRONMENT:

INTEREST RATES:
- Federal Funds Rate: Current monetary policy stance affects discount rates
- 10-Year Treasury: Key benchmark for long-term interest rates
- Credit spreads: Corporate bond spreads indicate credit risk

INFLATION INDICATORS:
- CPI: Consumer price inflation affects purchasing power
- PPI: Producer price inflation affects corporate costs
- Core inflation: Excludes volatile food and energy

GROWTH INDICATORS:
- GDP Growth: Overall economic expansion rate
- Employment: Unemployment rate and job creation
- Consumer Confidence: Spending intentions

MARKET CONDITIONS:
- VIX: Volatility index indicating market fear/complacency
- Dollar Index: USD strength affects multinational companies
- Commodity prices: Input costs for various industries

Note: For real-time data, integrate with FRED API, Alpha Vantage, or similar services.
        """
        return economic_summary

    except Exception as e:
        return f"Error fetching economic data: {e}"

@tool
def portfolio_analysis_tool(tickers: str, weights: str = "equal") -> str:
    """
    Portfolio analysis tool for portfolio-level analysis and optimization.
    Analyzes portfolio composition, diversification, correlation,
    and suggests optimization strategies.
    """
    global _smart_cache
    try:
        # Normalize and dedupe ticker list (preserves order)
        ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        seen = set()
        ticker_list = [t for t in ticker_list if not (t in seen or seen.add(t))]

        if len(ticker_list) < 2:
            return "Portfolio analysis requires at least 2 stocks"

        # Fetch data through the shared cache
        data = {}
        failed_tickers = []
        for ticker in ticker_list:
            cached = _smart_cache.get_ticker_data(ticker)
            if cached is None or cached.get('history') is None:
                failed_tickers.append(ticker)
                continue

            hist = _smart_cache.slice_history(cached['history'], '1y')
            if hist is not None and not hist.empty:
                data[ticker] = hist['Close']
            else:
                failed_tickers.append(ticker)

        if failed_tickers:
            print(f"⚠️ Could not get history for: {failed_tickers}")

        if len(data) < 2:
            return (f"Could not fetch sufficient data for portfolio analysis. "
                    f"Got data for {list(data.keys())}, failed: {failed_tickers}")

        # Filter ticker list to successful fetches so weights still align
        ticker_list = [t for t in ticker_list if t in data]

        # Build returns DataFrame
        portfolio_df = pd.DataFrame(data)
        portfolio_df = portfolio_df.dropna()
        returns = portfolio_df.pct_change().dropna()

        # Correlation matrix
        correlation_matrix = returns.corr()

        # Parse weights with explicit error reporting
        weight_warnings = []

        if weights == "equal":
            weight_values = np.array([1.0 / len(ticker_list)] * len(ticker_list))
        else:
            try:
                weight_list = [float(w.strip()) for w in weights.split(',')]
            except ValueError as e:
                weight_warnings.append(
                    f"⚠️ Could not parse weights '{weights}' ({e}). Falling back to equal weights."
                )
                weight_list = None

            if weight_list is None:
                weight_values = np.array([1.0 / len(ticker_list)] * len(ticker_list))
            elif len(weight_list) != len(ticker_list):
                weight_warnings.append(
                    f"⚠️ Weight count mismatch: got {len(weight_list)} weights for "
                    f"{len(ticker_list)} tickers. Falling back to equal weights."
                )
                weight_values = np.array([1.0 / len(ticker_list)] * len(ticker_list))
            else:
                weight_array = np.array(weight_list, dtype=float)
                weight_sum = weight_array.sum()

                if weight_sum <= 0:
                    weight_warnings.append(
                        f"⚠️ Weight sum is {weight_sum:.4f} (must be positive). "
                        f"Falling back to equal weights."
                    )
                    weight_values = np.array([1.0 / len(ticker_list)] * len(ticker_list))
                else:
                    weight_values = weight_array / weight_sum
                    if not np.isclose(weight_sum, 1.0, atol=0.01):
                        weight_warnings.append(
                            f"ℹ️ Weights summed to {weight_sum:.4f}, normalized to 1.0."
                        )

        # Print warnings immediately so they're visible during runs
        for warning in weight_warnings:
            print(warning)

        # Individual stock statistics
        individual_returns = returns.mean() * 252
        individual_volatility = returns.std() * np.sqrt(252)

        # Portfolio-level statistics
        portfolio_return = np.sum(individual_returns * weight_values)
        portfolio_variance = np.dot(weight_values.T, np.dot(returns.cov() * 252, weight_values))
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Sharpe ratios with dynamic risk-free rate
        risk_free_rate = _get_risk_free_rate()
        individual_sharpe = (individual_returns - risk_free_rate) / individual_volatility
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

        analysis = f"""
PORTFOLIO ANALYSIS:

COMPOSITION:
{', '.join([f"{ticker}: {weight:.1%}" for ticker, weight in zip(ticker_list, weight_values)])}
"""

        if weight_warnings:
            analysis += "\nWEIGHT NOTES:\n" + "\n".join(weight_warnings) + "\n"

        analysis += f"""
PORTFOLIO METRICS:
- Expected Return: {portfolio_return:.1%}
- Volatility: {portfolio_volatility:.1%}
- Sharpe Ratio: {portfolio_sharpe:.2f} (risk-free rate: {risk_free_rate:.2%})

CORRELATION MATRIX:
{correlation_matrix.round(3).to_string()}

INDIVIDUAL STOCK METRICS:
{'Ticker':<8} {'Return':<8} {'Volatility':<12} {'Sharpe':<8}
{'-'*40}
"""
        for ticker in ticker_list:
            if ticker in individual_returns.index:
                analysis += f"{ticker:<8} {individual_returns[ticker]:.1%}   {individual_volatility[ticker]:.1%}      {individual_sharpe[ticker]:.2f}\n"

        # Diversification analysis
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()

        pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    pairs.append(f"{correlation_matrix.columns[i]}-{correlation_matrix.columns[j]}: {corr:.2f}")
        high_corr_pairs = pairs[:3] if pairs else ["None above 0.7 threshold"]

        analysis += f"""
DIVERSIFICATION METRICS:
- Average Correlation: {avg_correlation:.3f}
- Diversification Benefit: {'High' if avg_correlation < 0.3 else 'Medium' if avg_correlation < 0.7 else 'Low'}

RECOMMENDATIONS:
- {'Consider adding uncorrelated assets' if avg_correlation > 0.7 else 'Good diversification level'}
- Highest correlation pairs: {high_corr_pairs}
        """

        return analysis

    except Exception as e:
        return f"Error in portfolio analysis: {e}"

# --- Enhanced Agent Definitions ---

def create_enhanced_agents(llm_configs):
    """PROFESSIONAL: 4-agent system with strategic LLM assignment"""
    
    scrape_tool = ScrapeWebsiteTool()
    search_tool = SerperDevTool()
    agents = {}

    agents['market_data_analyst'] = Agent(
        role="Senior Technical Analyst",
        goal="Institutional-grade technical analysis with advanced pattern recognition and market microstructure insights.",
        backstory="Senior technical analyst with 15+ years institutional trading experience, expert in multi-timeframe analysis and market structure.",
        verbose=True,
        allow_delegation=False,
        llm=llm_configs['technical'],
        tools=[enhanced_financial_data_tool, search_tool]
    )

    agents['sentiment_analyst'] = Agent(
        role="Senior Market Research Analyst",
        goal="Comprehensive sentiment analysis integrating market psychology, fundamental research, and macroeconomic factors.",
        backstory="Senior research analyst combining behavioral finance expertise with fundamental analysis and economic research capabilities.",
        verbose=True,
        allow_delegation=False,
        llm=llm_configs['sentiment'],
        tools=[market_sentiment_tool, search_tool, economic_data_tool, scrape_tool]
    )

    agents['risk_manager'] = Agent(
        role="Senior Risk Management Director",
        goal="Institutional-grade risk assessment with quantitative modeling and portfolio optimization expertise.",
        backstory="Risk management director with expertise in quantitative modeling, stress testing, and institutional portfolio risk management.",
        verbose=True,
        allow_delegation=False,
        llm=llm_configs['risk'],
        tools=[enhanced_financial_data_tool, portfolio_analysis_tool, economic_data_tool]
    )

    agents['portfolio_manager'] = Agent(
        role="Chief Investment Officer & Senior Portfolio Manager",
        goal="Final investment decisions with comprehensive strategy integration, portfolio management, and institutional execution expertise.",
        backstory="Chief Investment Officer with institutional experience in investment strategy, portfolio optimization, and capital allocation decisions.",
        verbose=True,
        allow_delegation=False,
        llm=llm_configs['portfolio'],
        tools=[portfolio_analysis_tool, enhanced_financial_data_tool]
    )

    return agents

# --- Enhanced Task Definitions ---

def create_enhanced_tasks(agents):
    """PROFESSIONAL: Enhanced 4-agent task definitions"""
    
    tasks = {}

    # Remove macro_analysis and strategy_development tasks
    # Keep only these 4 tasks:

    tasks['technical_analysis'] = Task(
        description="""
        INSTITUTIONAL-GRADE TECHNICAL ANALYSIS for {stock_selection}:
        1. Use enhanced_financial_data_tool({stock_selection}, period="2y", include_indicators=True)
        2. Multi-timeframe analysis (daily, weekly, monthly trend alignment)
        3. Advanced pattern recognition and market structure analysis
        4. Comprehensive indicator analysis (RSI, MACD, Bollinger Bands, volume, momentum)
        5. Support/resistance mapping with historical validation
        6. Volatility regime analysis and market conditions assessment
        7. Sector relative strength and institutional flow analysis
        8. Specific entry/exit levels with risk/reward calculations
        
        Provide institutional-quality technical analysis with detailed reasoning.
        """,
        expected_output="""Professional Technical Analysis Report:
        - Multi-timeframe trend analysis and convergence/divergence signals
        - Precise support/resistance levels with historical significance
        - Complete technical indicator interpretation with signal confirmation
        - Volume analysis and institutional flow detection
        - Volatility assessment and market regime classification
        - Specific trading levels with risk/reward ratios
        - Sector analysis and relative performance metrics""",
        agent=agents['market_data_analyst']
    )

    tasks['comprehensive_sentiment'] = Task(
        description="""
        COMPREHENSIVE MARKET ANALYSIS for {stock_selection}:
        1. Use market_sentiment_tool({stock_selection}, days_back=60)
        2. Research earnings, guidance updates, and analyst revisions
        3. Use economic_data_tool for macroeconomic factor assessment
        4. Evaluate news sentiment and institutional positioning
        5. Identify catalysts and event timeline analysis
        6. Assess competitive positioning and sector trends
        7. Analyze policy impacts and regulatory considerations
        
        Integrate sentiment, fundamental, and macro analysis comprehensively.
        """,
        expected_output="""Comprehensive Market Research Report:
        - Market sentiment direction with supporting evidence and momentum analysis
        - Analyst consensus evolution and rating change rationale  
        - Macroeconomic factor impact assessment and policy implications
        - Catalyst timeline with probability-weighted impact analysis
        - Competitive dynamics and market share trend analysis
        - Institutional positioning and smart money flow indicators""",
        agent=agents['sentiment_analyst']
    )

    tasks['risk_assessment'] = Task(
        description="""
        INSTITUTIONAL RISK MANAGEMENT ANALYSIS for {stock_selection}:
        1. Comprehensive volatility analysis and risk metric calculations
        2. Correlation assessment with market, sector, and economic factors
        3. Stress testing and tail risk scenario modeling
        4. Position sizing optimization for {risk_tolerance} risk profile
        5. Portfolio impact analysis and diversification assessment
        6. Risk monitoring framework with specific trigger development
        7. Maximum drawdown analysis and capital preservation strategies
        """,
        expected_output="""Professional Risk Management Report:
        - Quantitative risk metrics (VaR, CVaR, volatility, correlation analysis)
        - Position sizing recommendations with mathematical risk justification
        - Comprehensive scenario analysis across market environments
        - Risk monitoring framework with specific adjustment triggers
        - Portfolio integration impact and diversification benefit analysis
        - Capital preservation strategies and maximum loss parameters""",
        agent=agents['risk_manager'],
        context=[tasks['technical_analysis'], tasks['comprehensive_sentiment']]
    )

    tasks['portfolio_integration'] = Task(
        description="""
        CHIEF INVESTMENT OFFICER FINAL DECISION for {stock_selection}:
        1. Synthesize technical, market research, and risk analyses comprehensively
        2. Develop complete investment strategy for {trading_strategy_preference}
        3. Determine optimal allocation timing for {initial_capital} portfolio
        4. Create detailed implementation roadmap with contingency planning
        5. Provide institutional-grade BUY/SELL/HOLD recommendation
        6. Design comprehensive monitoring and portfolio management framework
        7. Include multiple scenario planning and systematic exit strategies
        """,
        expected_output="""Chief Investment Officer Decision Report:
        - EXECUTIVE SUMMARY: Clear BUY/SELL/HOLD with confidence level and comprehensive rationale
        - INVESTMENT THESIS: Detailed strategy integration and conviction assessment
        - IMPLEMENTATION PLAN: Specific timing, sizing, and execution methodology  
        - RISK FRAMEWORK: Complete risk management with scenario contingencies
        - MONITORING SYSTEM: Professional review schedule and adjustment triggers
        - PORTFOLIO INTEGRATION: Capital allocation impact and optimization assessment""",
        agent=agents['portfolio_manager'],
        context=[tasks['technical_analysis'], tasks['comprehensive_sentiment'], tasks['risk_assessment']]
    )

    return tasks

# --- Enhanced Execution Framework ---

class EnhancedTradingSystem:
    """Enhanced trading system with Claude Sonnet 4 and comprehensive analysis capabilities"""
    
    def __init__(self, api_config: APIConfig):
        self.api_config = api_config
        
        # Validate API configuration
        if not api_config.validate():
            raise ValueError("Invalid API configuration. Please set valid API keys.")
            
        # Check Claude credits and connection
        if not check_claude_credits(api_config):
            raise ValueError("Claude API connection failed. Check your API key and credits.")
            
        self.llm_configs, self.manager_llm = setup_claude_llms(api_config)

        if not self.llm_configs or not self.manager_llm:
            raise ValueError("Failed to initialize professional LLM configuration.")
    
        self.agents = create_enhanced_agents(self.llm_configs)
        self.tasks = create_enhanced_tasks(self.agents)
        self.api_call_count = 0
        self.delay_count = 0


    def professional_delay(self, phase: str, agent_name: str = None, context_size: int = 0):
        """Professional-grade delay system for zero-failure reliability"""

        # Phase-specific delays for optimal reliability
        phase_delays = {
            'initialization': 10.0,
            'technical_analysis': 15.0,
            'sentiment_analysis': 20.0,
            'risk_assessment': 25.0,
            'final_decision': 30.0,
            'completion': 5.0
        }

        base_delay = phase_delays.get(phase, self.api_config.rate_limit_delay)

        # Additional context processing delay
        if context_size > 15000:
            context_delay = self.api_config.context_processing_delay * 2
        elif context_size > 8000:
            context_delay = self.api_config.context_processing_delay
        else:
            context_delay = 0

        total_delay = base_delay + context_delay + random.uniform(1, 3)

        print(f"Professional rate limiting: {phase}")
        if agent_name:
            print(f"Processing agent: {agent_name}")
        print(f"Waiting {total_delay:.1f}s for reliable execution...")

        time.sleep(total_delay)
        self.delay_count += 1
        
    def create_crew(self, analysis_mode: str = "smart"):
        """PROFESSIONAL: Create 4-agent crew with strategic model assignment"""

        if analysis_mode == "quick":
        # Single agent for quick analysis
            selected_agents = [self.agents['market_data_analyst']]
            selected_tasks = [self.tasks['technical_analysis']]
        
        elif analysis_mode == "smart":
        # 3 core agents for balanced analysis
            selected_agents = [
                self.agents['market_data_analyst'],
                self.agents['sentiment_analyst'], 
                self.agents['portfolio_manager']
            ]
            selected_tasks = [
                self.tasks['technical_analysis'],
                self.tasks['comprehensive_sentiment'],
                self.tasks['portfolio_integration']
            ]
        
        elif analysis_mode == "comprehensive":
        # All 4 agents for maximum professional analysis
            selected_agents = [
                self.agents['market_data_analyst'],
                self.agents['sentiment_analyst'],
                self.agents['risk_manager'],
                self.agents['portfolio_manager']
            ]
            selected_tasks = [
                self.tasks['technical_analysis'],
                self.tasks['comprehensive_sentiment'], 
                self.tasks['risk_assessment'],
                self.tasks['portfolio_integration']
            ]
        else:
            raise ValueError("Analysis mode must be 'quick', 'smart', or 'comprehensive'")

        return Crew(
            agents=selected_agents,
            tasks=selected_tasks,
            manager_llm=self.manager_llm,
            process=Process.sequential,  # Sequential for reliability
            verbose=True,
            max_rpm=30,  # API rate limit protection
            max_execution_time=1800  # 30 minutes maximum
        )

    def fallback_single_agent_analysis(self, stock_selection: str, inputs: dict):
        """Fallback: Single comprehensive agent to avoid rate limits"""

        print("🔄 FALLBACK: Single-agent comprehensive analysis")

        # Use the portfolio LLM since this fallback needs to do everything end-to-end.
        # Instantiate search_tool locally — matches the pattern in create_enhanced_agents.
        fallback_llm = self.llm_configs['portfolio']
        search_tool = SerperDevTool()

        comprehensive_agent = Agent(
            role="Senior Financial Analyst",
            goal=f"Complete analysis of {stock_selection} for {inputs['trading_strategy_preference']}",
            backstory="Expert analyst with technical, sentiment, and risk expertise.",
            verbose=True,
            allow_delegation=False,
            llm=fallback_llm,
            tools=[enhanced_financial_data_tool, market_sentiment_tool, search_tool]
        )

        # Single comprehensive task
        comprehensive_task = Task(
            description=f"""
            Complete analysis for {stock_selection}:

            1. TECHNICAL: Use enhanced_financial_data_tool("{stock_selection}", period="1y", include_indicators=True)
            2. SENTIMENT: Use market_sentiment_tool("{stock_selection}", days_back=30)
            3. SEARCH: Look up recent news and analyst updates
            4. INTEGRATION: Provide BUY/SELL/HOLD recommendation
            5. LEVELS: Give specific entry/exit levels and stop-losses
            """,
            expected_output=f"""
            {stock_selection} Comprehensive Analysis:
            - Technical analysis with current levels
            - Sentiment assessment and catalysts
            - Clear BUY/SELL/HOLD recommendation
            - Specific entry/exit levels
            - Risk management strategy
            """,
            agent=comprehensive_agent
        )

        try:
            time.sleep(10)  # Wait for rate limit reset

            crew = Crew(
                agents=[comprehensive_agent],
                tasks=[comprehensive_task],
                process=Process.sequential,
                verbose=True
            )

            result = crew.kickoff(inputs=inputs)
            print("✅ Fallback analysis successful!")
            return result

        except Exception as e:
            print(f"❌ Fallback also failed: {e}")
            return None

    def smart_analyze_stock(self, stock_selection: str, initial_capital: str = "100000",
                           risk_tolerance: str = "Medium", 
                           trading_strategy_preference: str = "Swing trading",
                           analysis_mode: str = "smart",
                           save_to_file: bool = True):
        """PROFESSIONAL: Smart analysis with strategic delays"""
    
        inputs = {
            'stock_selection': stock_selection,
            'initial_capital': initial_capital,
            'risk_tolerance': risk_tolerance,
            'trading_strategy_preference': trading_strategy_preference
        }

        print(f"Starting Professional Analysis for {stock_selection}")
        print(f"Mode: {analysis_mode} with strategic LLM assignment")
        print(f"Model: {self.api_config.claude_model}")
        print(f"Expected duration: 12-15 minutes for zero-failure reliability")
        print("=" * 60)

        start_time = time.time()

        try:
        # PHASE 1: System initialization
            self.professional_delay('initialization')
        
        # Create crew with strategic agents
            agents = create_enhanced_agents(self.llm_configs)
            tasks = create_enhanced_tasks(agents)

            crew = Crew(
                agents=list(agents.values()),
                tasks=list(tasks.values()),
                manager_llm=self.manager_llm,
                process=Process.sequential,
                verbose=True,
                max_rpm=30,
                max_execution_time=1800
            )

            print("Professional crew assembled with strategic model assignment")

        # PHASE 2: Execute analysis with professional delays
            print("\nExecuting professional-grade analysis phases...")
        
        # Before technical analysis
            self.professional_delay('technical_analysis', 'Market Data Analyst')
        
        # Before sentiment analysis  
            self.professional_delay('sentiment_analysis', 'Sentiment Analyst')
        
        # Before risk assessment (only for comprehensive mode)
            if analysis_mode == "comprehensive":
                self.professional_delay('risk_assessment', 'Risk Manager')
        
        # Before final decision
            self.professional_delay('final_decision', 'Portfolio Manager')

        # Execute the crew
            result = crew.kickoff(inputs=inputs)

        # PHASE 3: Completion
            self.professional_delay('completion')

            end_time = time.time()
            duration = (end_time - start_time) / 60

            print("=" * 60)
            print(f"Professional Analysis Complete!")
            print(f"Duration: {duration:.1f} minutes")
            print(f"API calls made: {self.api_call_count}")
            print(f"Estimated cost: ${(self.api_call_count * 0.15):.2f}")

        # Save to file if requested
            if save_to_file and result:
                saved_file = self.save_analysis_to_file(result, stock_selection, f"professional_{analysis_mode}")
                if saved_file:
                    print(f"Professional analysis saved to: {saved_file}")

            return result

        except Exception as e:
            print(f"Professional analysis failed: {e}")
            print("Attempting single-agent fallback...")
            return self.fallback_single_agent_analysis(stock_selection, inputs)

    def save_analysis_to_file(self, result, stock_symbol: str, analysis_type: str = "comprehensive"):
        """Save analysis results to timestamped files"""
        if result is None:
            print("❌ No result to save")
            return None

        try:
            # Create results directory if it doesn't exist
            import os
            results_dir = "trading_analysis_results"
            os.makedirs(results_dir, exist_ok=True)

            # Generate timestamped filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{results_dir}/{stock_symbol}_{analysis_type}_{timestamp}.md"

            # Extract content
            content = result.raw if hasattr(result, 'raw') else str(result)

            # Create formatted markdown
            markdown_content = f"""# Financial Analysis Report: {stock_symbol}
**Analysis Type:** {analysis_type.title()}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Model:** {self.api_config.claude_model}  
**API Calls:** {self.api_call_count}  

---

{content}

---
*Generated by Enhanced Multi-Agent Trading System*
"""

            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"✅ Analysis saved to: {filename}")
            return filename

        except Exception as e:
            print(f"❌ Failed to save analysis: {e}")
            return None

    def analyze_stock(self, stock_selection: str, initial_capital: str = "100000",
                     risk_tolerance: str = "Medium", 
                     trading_strategy_preference: str = "Swing trading",
                     analysis_mode: str = "comprehensive",
                     save_to_file: bool = True):
        """Run enhanced stock analysis with Claude Sonnet 4"""
        
        inputs = {
            'stock_selection': stock_selection,
            'initial_capital': initial_capital,
            'risk_tolerance': risk_tolerance,
            'trading_strategy_preference': trading_strategy_preference
        }
        
        crew = self.create_crew(analysis_mode)
        
        print(f"🚀 Starting Enhanced Trading Analysis for {stock_selection}")
        print(f"📊 Mode: {analysis_mode.title()}")
        print(f"🧠 Model: {self.api_config.claude_model}")
        print(f"⏱️ Expected duration: {2 if analysis_mode == 'comprehensive' else 1}-{3 if analysis_mode == 'comprehensive' else 1.5} minutes")
        print(f"📏 Max output: {self.api_config.max_output_tokens} tokens per response")
        print(f"📁 Save to file: {'Yes' if save_to_file else 'No'}")
        print(f"🔧 Inputs: {inputs}")
        print("=" * 60)
        
        start_time = time.time()
        try:
            # Initial delay before starting
            self.professional_delay('initialization')
            
            result = crew.kickoff(inputs=inputs)
            
            end_time = time.time()
            duration = (end_time - start_time) / 60
            
            print("=" * 60)
            print(f"✅ Analysis Complete!")
            print(f"⏱️ Duration: {duration:.1f} minutes")
            print(f"📞 API calls made: {self.api_call_count}")
            print(f"💰 Estimated cost: ${(self.api_call_count * 0.12):.2f}")
            print(f"📊 Check console.anthropic.com for exact usage")
            
            # Save to file if requested
            if save_to_file and result:
                saved_file = self.save_analysis_to_file(result, stock_selection, analysis_mode)
                if saved_file:
                    print(f"📁 Analysis saved to: {saved_file}")
            
            return result
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            print(f"💡 Troubleshooting:")
            print(f"• Check Claude API key and credits")
            print(f"• Verify Serper API key is valid")
            print(f"• Check console.anthropic.com for rate limits")
            print(f"• Try: pip install --upgrade anthropic crewai")
            return None

    def analyze_stock_with_fallback(self, stock_list: List[str], initial_capital: str = "100000",
                                  risk_tolerance: str = "Medium", 
                                  trading_strategy_preference: str = "Swing trading",
                                  analysis_mode: str = "comprehensive",
                                  max_attempts: int = 3):
        """Analyze stocks with automatic fallback to alternative stocks"""
        
        print(f"🎯 Multi-Stock Analysis with Fallback")
        print(f"📋 Stock candidates: {stock_list[:max_attempts]}")
        print(f"🔄 Max attempts: {max_attempts}")
        print("=" * 60)
        
        for i, stock_ticker in enumerate(stock_list[:max_attempts], 1):
            print(f"\n🔍 Attempt {i}/{min(max_attempts, len(stock_list))}: Analyzing {stock_ticker}")
            
            try:
                # First, validate stock data
                test_data = enhanced_financial_data_tool(stock_ticker, period="1mo")
                if "Error" in test_data:
                    print(f"⚠️ {stock_ticker}: Data not available - {test_data}")
                    continue
                
                # Run full analysis
                result = self.analyze_stock(
                    stock_selection=stock_ticker,
                    initial_capital=initial_capital,
                    risk_tolerance=risk_tolerance,
                    trading_strategy_preference=trading_strategy_preference,
                    analysis_mode=analysis_mode
                )
                
                if result is not None:
                    print(f"✅ SUCCESS: Analysis completed for {stock_ticker}")
                    return result, stock_ticker
                else:
                    print(f"❌ {stock_ticker}: Analysis failed, trying next stock...")
                    
            except Exception as e:
                print(f"❌ {stock_ticker}: Error - {e}")
                continue
        
        print(f"💔 All stock analyses failed after {min(max_attempts, len(stock_list))} attempts")
        return None, None

    def analyze_portfolio(self, tickers: List[str], weights: Optional[List[float]] = None):
        """Analyze multiple stocks as a portfolio"""
        ticker_string = ','.join(tickers)
        
        if weights:
            weight_string = ','.join([str(w) for w in weights])
        else:
            weight_string = "equal"
        
        try:
            return portfolio_analysis_tool(ticker_string, weight_string)
        except Exception as e:
            print(f"Portfolio analysis error: {e}")
            return f"Portfolio Analysis for {ticker_string}: Analysis framework ready, data processing encountered an issue."

class SingleAgentTradingSystem:
    """Combined system with all enhancements and original robustness"""
    
    def __init__(self, api_config: APIConfig):
        self.api_config = api_config
        
        # Setup environment variables
        import os
        os.environ["ANTHROPIC_API_KEY"] = api_config.claude_api_key
        os.environ["SERPER_API_KEY"] = api_config.serper_api_key
        
        # Validate configuration
        if not api_config.validate():
            raise ValueError("Invalid API configuration")
        
        # Check connection
        if not check_claude_credits(api_config):
            raise ValueError("Claude API connection failed")
        
        # Create optimized LLMs for different use cases
        self.create_optimized_llms()
        
        # Strategy templates from optimized version
        self.analysis_templates = {
            "swing_trading": {
                "period": "1y",
                "focus": "medium-term trends, support/resistance levels",
                "timeframe": "1-6 months holding period",
                "risk_tolerance": "medium"
            },
            "day_trading": {
                "period": "6mo", 
                "focus": "short-term momentum, intraday levels",
                "timeframe": "intraday to 1 week",
                "risk_tolerance": "high"
            },
            "value_investing": {
                "period": "2y",
                "focus": "fundamental value, long-term trends", 
                "timeframe": "6 months to 2 years",
                "risk_tolerance": "low"
            },
            "momentum": {
                "period": "1y",
                "focus": "trend continuation, breakout patterns",
                "timeframe": "2-12 weeks",
                "risk_tolerance": "medium-high"
            },
            "custom": {  # Added for flexibility
                "period": "1y",
                "focus": "user-defined analysis parameters",
                "timeframe": "flexible",
                "risk_tolerance": "user-defined"
            }
        }
        
        self.api_call_count = 0
        print("✅ Ultimate Single-Agent Trading System initialized")
        print("🛡️ Full fallback protection + all enhancements")
        print("🎯 Supports any stock ticker and trading strategy")
    
    def create_optimized_llms(self):
        """Create multiple LLM configurations for different scenarios"""
        # Comprehensive analysis LLM
        self.llm_comprehensive = LLM(
            model="anthropic/claude-sonnet-4-20250514",
            api_key=self.api_config.claude_api_key,
            max_tokens=8000,
            timeout=120,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        
        # Quick analysis LLM
        self.llm_quick = LLM(
            model="anthropic/claude-sonnet-4-20250514",
            api_key=self.api_config.claude_api_key,
            max_tokens=6000,
            timeout=60,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        
        # Minimal analysis LLM (for fallbacks)
        self.llm_minimal = LLM(
            model="anthropic/claude-sonnet-4-20250514",
            api_key=self.api_config.claude_api_key,
            max_tokens=4000,
            timeout=30,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate ticker symbol and get basic info"""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker.upper())
            info = stock.info
            
            if 'symbol' in info or 'shortName' in info or 'longName' in info:
                company_name = info.get('longName', info.get('shortName', ticker))
                print(f"✅ Ticker validated: {company_name} ({ticker.upper()})")
                return True
            else:
                print(f"⚠️ Ticker {ticker} might be invalid, but proceeding anyway...")
                return True  # Still try to analyze even if validation is uncertain
                
        except Exception as e:
            print(f"⚠️ Ticker validation warning for {ticker}: {e}")
            return True  # Proceed anyway, let the tools handle invalid tickers
    
    def analyze_stock_comprehensive(self, ticker: str, 
                                  initial_capital: str = "100000",
                                  risk_tolerance: str = "Medium",
                                  trading_strategy: str = "swing_trading",
                                  custom_focus: str = None,
                                  skip_validation: bool = False):
        """
        Comprehensive analysis with full fallback protection
        Merges the best of both original and optimized versions
        """
        
        ticker = ticker.upper().strip()
        
        # Validate ticker unless skipped
        if not skip_validation and not self.validate_ticker(ticker):
            print(f"⚠️ Proceeding with {ticker} despite validation issues...")
        
        # Get strategy template or use custom
        if trading_strategy not in self.analysis_templates:
            print(f"📝 Using custom strategy parameters for '{trading_strategy}'")
            template = self.analysis_templates["custom"]
            template["focus"] = custom_focus or f"{trading_strategy} strategy analysis"
        else:
            template = self.analysis_templates[trading_strategy]
        
        print(f"🚀 ULTIMATE COMPREHENSIVE ANALYSIS: {ticker}")
        print(f"📊 Strategy: {trading_strategy.replace('_', ' ').title()}")
        print(f"🎯 Focus: {custom_focus or template['focus']}")
        print(f"💰 Capital: ${initial_capital}")
        print(f"⚠️ Risk: {risk_tolerance}")
        print("=" * 60)
        
        # Try comprehensive analysis first
        try:
            return self._comprehensive_agent_analysis(
                ticker, trading_strategy, template, 
                initial_capital, risk_tolerance, custom_focus
            )
        except Exception as e:
            print(f"⚠️ Comprehensive analysis failed: {e}")
            if "request_too_large" in str(e).lower():
                print("📉 Falling back to quick analysis...")
                return self._quick_agent_analysis(ticker, trading_strategy, template)
            else:
                print("📉 Falling back to minimal analysis...")
                return self._minimal_agent_fallback(ticker, trading_strategy, template)
    
    def _comprehensive_agent_analysis(self, ticker, strategy, template, 
                                     initial_capital, risk_tolerance, custom_focus):
        """Full comprehensive analysis - from original + enhancements"""
        
        focus_area = custom_focus or template['focus']
        
        comprehensive_analyst = Agent(
            role=f"{strategy.replace('_', ' ').title()} Financial Analyst",
            goal=f"Complete {strategy.replace('_', ' ')} analysis of {ticker}",
            backstory=f"Expert financial analyst specializing in {strategy.replace('_', ' ')} strategies with deep market knowledge.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm_comprehensive,
            tools=[enhanced_financial_data_tool, market_sentiment_tool, SerperDevTool()]
        )
        
        analysis_task = Task(
            description=f"""
            Complete {strategy.replace('_', ' ')} analysis for {ticker}:
            
            STEP 1 - TECHNICAL ANALYSIS ({template['period']} timeframe):
            - Use enhanced_financial_data_tool("{ticker}", period="{template['period']}", include_indicators=True)
            - Focus on {focus_area}
            - Analyze price trends, support/resistance for {template['timeframe']}
            - Interpret RSI, MACD, moving averages, volume patterns
            - Assess volatility and market conditions
            
            STEP 2 - SENTIMENT & CATALYST ANALYSIS:
            - Use market_sentiment_tool("{ticker}", days_back=30)
            - Evaluate current market sentiment and analyst views
            - Research recent {ticker} news, earnings updates, analyst changes
            - Identify upcoming catalysts relevant to {template['timeframe']}
            
            STEP 3 - STRATEGIC MARKET RESEARCH:
            - Search for "{ticker} {strategy.replace('_', ' ')} analysis {template['timeframe']}"
            - Research sector trends and competitive positioning
            - Evaluate how current market conditions affect the strategy
            
            STEP 4 - FINAL {strategy.upper()} RECOMMENDATION:
            - Integrate technical + sentiment + research findings
            - Provide clear BUY/SELL/HOLD recommendation with confidence level
            - Give specific entry/exit levels and stop-losses
            - Calculate position sizing for ${initial_capital} portfolio
            - Design risk management for {risk_tolerance.lower()} risk tolerance
            
            Be specific, actionable, and thorough.
            """,
            expected_output=f"""
            {ticker} {strategy.replace('_', ' ').title()} Analysis Report:
            
            ## EXECUTIVE SUMMARY
            - **STRATEGY**: {strategy.replace('_', ' ').title()}
            - **RECOMMENDATION**: [BUY/SELL/HOLD]
            - **CONFIDENCE**: [High/Medium/Low]
            - **CURRENT PRICE**: $XXX.XX
            - **TARGET PRICE**: $XXX.XX (upside/downside: X%)
            - **STOP LOSS**: $XXX.XX
            - **POSITION SIZE**: X% of ${initial_capital} portfolio
            - **TIME HORIZON**: {template['timeframe']}
            
            ## TECHNICAL ANALYSIS
            - Price Trend & Momentum
            - Key Support/Resistance Levels
            - Technical Indicators Assessment
            - Volume Analysis
            - Volatility Metrics
            
            ## SENTIMENT & CATALYSTS
            - Market Sentiment Score
            - Analyst Consensus
            - Upcoming Events & Catalysts
            - News Impact Assessment
            
            ## TRADING STRATEGY
            - Entry Plan & Timing
            - Exit Strategy & Targets
            - Risk Management Rules
            - Position Monitoring Framework
            
            ## RISK ASSESSMENT
            - Risk/Reward Ratio
            - Scenario Analysis
            - Portfolio Impact
            - Key Risk Factors
            """,
            agent=comprehensive_analyst
        )
        
        return self._execute_with_fallbacks(
            comprehensive_analyst, analysis_task, ticker, 
            strategy, "comprehensive", initial_capital
        )
    
    def _quick_agent_analysis(self, ticker, strategy, template):
        """Quick analysis fallback - enhanced version"""
        
        print(f"⚡ Quick analysis mode for {ticker}")
        
        quick_analyst = Agent(
            role="Financial Analyst",
            goal=f"Quick {ticker} analysis for {strategy.replace('_', ' ')}",
            backstory="Efficient financial analyst providing rapid insights.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm_quick,
            tools=[enhanced_financial_data_tool, market_sentiment_tool]
        )
        
        quick_task = Task(
            description=f"""
            Quick {ticker} analysis for {strategy.replace('_', ' ')} strategy:
            1. Use enhanced_financial_data_tool("{ticker}", period="{template['period']}")
            2. Use market_sentiment_tool("{ticker}", days_back=15)
            3. Focus on {template['focus']}
            4. Provide clear BUY/SELL/HOLD with key levels
            """,
            expected_output=f"""
            {ticker} Quick Analysis:
            - Current Price & Trend
            - Key Support/Resistance
            - Sentiment Overview
            - RECOMMENDATION: BUY/SELL/HOLD
            - Entry: $XXX.XX
            - Target: $XXX.XX
            - Stop: $XXX.XX
            """,
            agent=quick_analyst
        )
        
        return self._execute_with_fallbacks(
            quick_analyst, quick_task, ticker, 
            strategy, "quick", "100000"
        )
    
    def _minimal_agent_fallback(self, ticker, strategy, template):
        """Minimal analysis fallback - from original"""
        
        print(f"🔧 Minimal analysis fallback for {ticker}")
        
        minimal_analyst = Agent(
            role="Analyst",
            goal=f"Basic {ticker} assessment",
            backstory="Financial analyst.",
            verbose=True,
            llm=self.llm_minimal,
            tools=[enhanced_financial_data_tool]
        )
        
        minimal_task = Task(
            description=f"Use enhanced_financial_data_tool('{ticker}', period='6mo') and provide BUY/SELL/HOLD with price levels.",
            expected_output=f"{ticker}: Recommendation, current price, entry, target, stop loss",
            agent=minimal_analyst
        )
        
        return self._execute_with_fallbacks(
            minimal_analyst, minimal_task, ticker, 
            strategy, "minimal", "100000"
        )
    
    def _execute_with_fallbacks(self, agent, task, ticker, strategy, mode, initial_capital):
        """Execute analysis with complete fallback chain"""
        
        try:
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            start_time = time.time()
            result = crew.kickoff()
            duration = (time.time() - start_time) / 60
            
            print("=" * 60)
            print(f"✅ {mode.title()} Analysis Complete!")
            print(f"⏱️ Duration: {duration:.1f} minutes")
            
            # Save results
            self.save_analysis(result, ticker, strategy, mode)
            
            return result
            
        except Exception as e:
            print(f"⚠️ {mode.title()} analysis failed: {e}")
            
            # Fallback chain
            if mode == "comprehensive":
                print("📉 Trying quick analysis...")
                return self._quick_agent_analysis(ticker, strategy, 
                                                self.analysis_templates.get(strategy, self.analysis_templates["custom"]))
            elif mode == "quick":
                print("📉 Trying minimal analysis...")
                return self._minimal_agent_fallback(ticker, strategy,
                                                  self.analysis_templates.get(strategy, self.analysis_templates["custom"]))
            elif mode == "minimal":
                print("🛠️ Trying direct tool analysis...")
                return self.direct_tool_analysis(ticker, strategy)
            else:
                return None
    
    def direct_tool_analysis(self, ticker: str, strategy: str = None):
        """Ultimate fallback: Use tools directly without agents - from original"""
        
        print(f"🛠️ DIRECT TOOL ANALYSIS: {ticker}")
        print("🔧 Bypassing agents, using tools directly")
        
        try:
            # Get template for period
            template = self.analysis_templates.get(strategy, self.analysis_templates["custom"])
            
            # Get data directly
            technical_data = enhanced_financial_data_tool(
                ticker, 
                period=template['period'], 
                include_indicators=True
            )
            sentiment_data = market_sentiment_tool(ticker, days_back=30)
            
            # Format direct analysis
            analysis = f"""
DIRECT TOOL ANALYSIS FOR {ticker}:
{'='*50}

TECHNICAL DATA ({template['period']} period):
{technical_data}

SENTIMENT DATA:
{sentiment_data}

ANALYSIS SUMMARY:
- Strategy: {strategy or 'General'}
- Focus: {template['focus']}
- Timeframe: {template['timeframe']}

NOTE: This is raw data from direct tool access. Agent-based analysis was unavailable due to API constraints.
Review the technical levels, sentiment indicators, and current price action to make your trading decision.
            """
            
            print("✅ Direct tool analysis successful!")
            self.save_analysis(analysis, ticker, strategy or "direct", "direct_tool")
            return analysis
            
        except Exception as e:
            print(f"❌ Even direct tool analysis failed: {e}")
            return f"Complete analysis failure for {ticker}. Error: {e}"
    
    def analyze_multiple_stocks(self, tickers: list, strategy: str = "swing_trading", 
                               mode: str = "quick", parallel: bool = False):
        """Analyze multiple stocks with smart batching - enhanced version"""
        
        print(f"📊 MULTI-STOCK ANALYSIS: {len(tickers)} stocks")
        print(f"🎯 Strategy: {strategy}, Mode: {mode}")
        print(f"⚡ Parallel: {'Yes' if parallel else 'No'}")
        
        results = {}
        successful = 0
        failed_tickers = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n🔍 Analyzing {ticker} ({i}/{len(tickers)})...")
            
            try:
                if mode == "comprehensive":
                    result = self.analyze_stock_comprehensive(
                        ticker, trading_strategy=strategy
                    )
                elif mode == "quick":
                    template = self.analysis_templates.get(strategy, self.analysis_templates["custom"])
                    result = self._quick_agent_analysis(ticker, strategy, template)
                elif mode == "minimal":
                    template = self.analysis_templates.get(strategy, self.analysis_templates["custom"])
                    result = self._minimal_agent_fallback(ticker, strategy, template)
                else:
                    result = self.direct_tool_analysis(ticker, strategy)
                
                if result:
                    results[ticker] = result
                    successful += 1
                    print(f"✅ {ticker} completed")
                else:
                    failed_tickers.append(ticker)
                    print(f"❌ {ticker} failed")
                
                # Add delay between stocks to avoid rate limits
                if i < len(tickers):
                    time.sleep(2 if mode == "minimal" else 3 if mode == "quick" else 5)
                    
            except Exception as e:
                print(f"❌ {ticker} error: {e}")
                failed_tickers.append(ticker)
                continue
        
        print(f"\n📊 MULTI-STOCK SUMMARY:")
        print(f"✅ Successful: {successful}/{len(tickers)}")
        if failed_tickers:
            print(f"❌ Failed: {failed_tickers}")
        print(f"📁 Results saved for: {list(results.keys())}")
        
        return results
    
    def analyze_portfolio(self, tickers: list, weights: list = None):
        """Portfolio analysis capability from original"""
        ticker_string = ','.join(tickers)
        weight_string = ','.join([str(w) for w in weights]) if weights else "equal"
        
        try:
            return portfolio_analysis_tool(ticker_string, weight_string)
        except Exception as e:
            print(f"Portfolio analysis error: {e}")
            return None
    
    def save_analysis(self, result, ticker: str, strategy: str = None, mode: str = None):
        """Enhanced save function with better organization"""
        try:
            import os
            from datetime import datetime

            os.makedirs("trading_analysis_results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Build filename
            filename_parts = [ticker]
            if strategy:
                filename_parts.append(strategy)
            if mode:
                filename_parts.append(mode)
            filename_parts.append(timestamp)

            filename = f"trading_analysis_results/{'_'.join(filename_parts)}.md"

            content = result if isinstance(result, str) else (result.raw if hasattr(result, 'raw') else str(result))

            markdown_content = f"""# {ticker} Analysis Report
**Strategy:** {strategy or 'General'}  
**Mode:** {mode or 'Unknown'}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**System:** Ultimate Single-Agent Trading System

---

{content}

---
*Generated by Ultimate Single-Agent Trading System*
"""

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"📁 Analysis saved to: {filename}")
            return filename

        except Exception as e:
            print(f"⚠️ Saving failed: {e}")
            return None


# Usage function
def run_ultimate_system():
    """Run the ultimate combined system"""
    
    system = SingleAgentTradingSystem(api_config)
    
    print("\n🎯 ULTIMATE TRADING SYSTEM OPTIONS:")
    print("1. Single Stock Comprehensive (Full fallback chain)")
    print("2. Multi-Stock Quick Screen")
    print("3. Custom Strategy Analysis")
    print("4. Portfolio Analysis")
    print("5. Batch Analysis (Popular stocks)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        ticker = input("Enter ticker (default: NVDA): ").strip() or "NVDA"
        capital = input("Enter capital (default: 100000): ").strip() or "100000"
        
        result = system.analyze_stock_comprehensive(
            ticker=ticker,
            initial_capital=capital,
            risk_tolerance="Medium",
            trading_strategy="swing_trading"
        )
        return result
        
    elif choice == "2":
        tickers = input("Enter tickers (comma-separated): ").strip()
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        
        results = system.analyze_multiple_stocks(
            ticker_list, 
            strategy="swing_trading", 
            mode="quick"
        )
        return results
        
    elif choice == "3":
        ticker = input("Enter ticker: ").strip().upper()
        strategy = input("Strategy (swing_trading/day_trading/value_investing/momentum/custom): ").strip()
        focus = input("Custom focus (optional): ").strip() or None
        
        result = system.analyze_stock_comprehensive(
            ticker=ticker,
            trading_strategy=strategy,
            custom_focus=focus
        )
        return result
        
    elif choice == "4":
        tickers = input("Enter portfolio tickers (comma-separated): ").strip()
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        weights_input = input("Enter weights (comma-separated, or blank for equal): ").strip()
        
        if weights_input:
            weights = [float(w) for w in weights_input.split(",")]
        else:
            weights = None
        
        result = system.analyze_portfolio(ticker_list, weights)
        return result
        
    elif choice == "5":
        tech_stocks = ["NVDA", "AAPL", "MSFT", "GOOGL", "META"]
        print(f"Analyzing: {tech_stocks}")
        
        results = system.analyze_multiple_stocks(
            tech_stocks, 
            strategy="momentum", 
            mode="quick"
        )
        return results
    
    else:
        print("Running default NVDA analysis...")
        return system.analyze_stock_comprehensive("NVDA")

# --- Usage Examples ---

def run_enhanced_examples():
    """UPDATED: Run enhanced system examples with smart rate limiting"""

    try:
        # Initialize system
        system = EnhancedTradingSystem(api_config)

        print("🎯 CLAUDE-POWERED ANALYSIS OPTIONS (RATE-LIMIT OPTIMIZED)")
        print("Choose your analysis type:")
        print("1. Smart NVDA Analysis (Rate-limit optimized, full quality)")
        print("2. Quick NVDA Test (Minimal tokens, fast execution)")
        print("3. Single-Agent Comprehensive (Avoids multi-agent overhead)")
        print("4. Portfolio Analysis Only (Correlation analysis)")
        print("5. Custom Stock Analysis (Enter your own ticker)")
        print("6. Single-Agent System (Recommended - No context issues)")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            print("=== SMART NVDA ANALYSIS (RATE-LIMIT OPTIMIZED) ===")
            return system.smart_analyze_stock(
                stock_selection="NVDA",
                initial_capital="250000",
                risk_tolerance="Medium-High",
                trading_strategy_preference="Growth momentum with technical confirmation",
                analysis_mode="smart",
                save_to_file=True
            )

        elif choice == "2":
            print("=== QUICK NVDA TEST (MINIMAL TOKENS) ===")
            return system.fallback_single_agent_analysis("NVDA", {
                'stock_selection': "NVDA",
                'initial_capital': "250000",
                'risk_tolerance': "Medium-High",
                'trading_strategy_preference': "Growth momentum"
            })

        elif choice == "3":
            print("=== SINGLE-AGENT COMPREHENSIVE ===")
            return system.fallback_single_agent_analysis("NVDA", {
                'stock_selection': "NVDA",
                'initial_capital': "250000",
                'risk_tolerance': "Medium-High",
                'trading_strategy_preference': "Swing trading"
            })

        elif choice == "4":
            print("=== PORTFOLIO ANALYSIS ONLY ===")
            portfolio_result = system.analyze_portfolio(
                tickers=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
            )
            print(portfolio_result)
            return portfolio_result

        elif choice == "5":
            print("=== CUSTOM STOCK ANALYSIS ===")
            custom_ticker = input("Enter stock ticker: ").strip().upper() or "AAPL"
            return system.smart_analyze_stock(
                stock_selection=custom_ticker,
                analysis_mode="smart",
                save_to_file=True
            )

        elif choice == "6":
            print("=== SINGLE-AGENT SYSTEM (RECOMMENDED) ===")
            single_system = SingleAgentTradingSystem(api_config)
            return single_system.analyze_stock_comprehensive("NVDA")

        else:
            print("❌ Invalid choice. Running Smart NVDA analysis by default.")
            return system.smart_analyze_stock(
                stock_selection="NVDA",
                analysis_mode="smart",
                save_to_file=True
            )

    except ValueError as e:
        print(f"Configuration Error: {e}")
        return None
    except Exception as e:
        print(f"System Error: {e}")
        return None

def run_single_analysis(stock_symbol: str = "AAPL", mode: str = "quick"):
    """Run a single analysis for command line usage"""
    try:
        system = EnhancedTradingSystem(api_config)
        
        print(f"🎯 SINGLE CLAUDE ANALYSIS: {stock_symbol} ({mode} mode)")
        print("=" * 50)
        
        result = system.analyze_stock(
            stock_selection=stock_symbol,
            analysis_mode=mode
        )
        
        return result
        
    except Exception as e:
        print(f"❌ Single analysis failed: {e}")
        return None


def _display_result(result):
    """Display a result regardless of its shape (CrewOutput, dict, str, or None)."""
    if result is None:
        return False

    print("\n" + "=" * 70)
    print("📈 CLAUDE ANALYSIS RESULTS:")
    print("=" * 70)

    if isinstance(result, dict):
        # Multi-stock results: ticker -> per-ticker result
        for ticker, per_ticker_result in result.items():
            print(f"\n--- {ticker} ---")
            content = (per_ticker_result.raw if hasattr(per_ticker_result, 'raw')
                       else str(per_ticker_result))
            try:
                display(Markdown(content))
            except (NameError, AttributeError):
                print(content)
    elif isinstance(result, str):
        # Plain string (e.g., portfolio analysis text)
        try:
            display(Markdown(result))
        except (NameError, AttributeError):
            print(result)
    else:
        # CrewOutput or similar object with .raw
        content = result.raw if hasattr(result, 'raw') else str(result)
        try:
            display(Markdown(content))
        except (NameError, AttributeError):
            print(content)

    return True

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 ENHANCED MULTI-AGENT TRADING SYSTEM - Claude Sonnet 4 Version")
    print("=" * 70)

    import os
    os.environ["ANTHROPIC_API_KEY"] = api_config.claude_api_key
    os.environ["SERPER_API_KEY"] = api_config.serper_api_key
    print("✅ Environment variables configured for LiteLLM")
    
    print("📋 REQUIREMENTS CHECK:")
    print("✅ API keys needed:")
    print("   • Claude API: https://console.anthropic.com (add credits)")
    print("   • Serper API: https://serper.dev (free tier)")
    print("✅ Yahoo Finance: Unlimited free (no API key)")
    print("✅ Rate limiting: Built-in 50 req/min protection")
    print("=" * 70)
    
    # Validate API configuration
    if not api_config.validate():
        print("\n❌ API CONFIGURATION REQUIRED")
        print("Please update the APIConfig class with your actual API keys:")
        print("• Replace [Claude-API-KEYS] with your Claude API key")
        print("• Replace [Serper-API-Keys] with your Serper API key")
        print("\n💡 Get Claude API key: console.anthropic.com")
        exit(1)
    
    # Command line options
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--check-credits":
        print("\n💰 CHECKING CLAUDE CREDITS...")
        check_claude_credits(api_config)
        exit(0)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--single-agent":
        stock = sys.argv[2] if len(sys.argv) > 2 else "NVDA"
        print(f"\n🚀 SINGLE-AGENT ANALYSIS: {stock}")
        single_system = SingleAgentTradingSystem(api_config)
        result = single_system.analyze_stock_comprehensive(stock)
        exit(0)

    
    # Main execution
    print("\n🎯 Starting Enhanced Trading System with Claude Sonnet 4...")
    print("⏱️ Rate limiting: 50 requests/minute (1.5 second delays)")
    print("🧠 Using Claude's advanced reasoning capabilities")
    print("📏 High token limits: 150K input, 8K output")
    print("💰 Estimated cost: ~$0.12 per comprehensive analysis")
    print("💡 Options:")
    print("   • python script.py --check-credits  (check Claude account)")
    print("   • python script.py --single AAPL quick  (single analysis)")
    
    print("\n🚀 SYSTEM SELECTION:")
    print("1. Enhanced Multi-Agent System (original)")
    print("2. Single-Agent System (recommended)")

    system_choice = input("Choose system (1, 2): ").strip()  # UPDATE

    if system_choice == "2":
        print("Using Single-Agent System...")
        result = run_ultimate_system()
    else:
        print("Using Enhanced Multi-Agent System...")
        result = run_enhanced_examples()

    # Display result
    if _display_result(result):
        print(f"\n💰 Check console.anthropic.com for exact costs")
        print(f"📞 API calls made: Variable based on analysis complexity")
    else:
        print("❌ System failed to initialize or run.")
        print("🔍 Check:")
        print("• Claude API key is correct")
        print("• You have credits in Claude account")
        print("• Serper API key is valid")
        print("• Internet connection is stable")
        print("• Try: python script.py --check-credits")

    print(f"\n📖 Useful links:")
    print(f"• Claude Console: https://console.anthropic.com")
    print(f"• Serper API: https://serper.dev/pricing")