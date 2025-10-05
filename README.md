# Agentic Financial Trading & Analysis System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![AI](https://img.shields.io/badge/AI-Claude%20Opus%204-purple.svg)

> This system employs specialized AI agents working collaboratively to provide comprehensive investment analysis. Each agent focuses on distinct expertise areas - technical patterns, market sentiment, risk assessment, and portfolio decisions - while a central coordinator ensures seamless information flow and strategic decision-making.

## Architecture

![Multi-Agent Architecture](multi-agent_architecture_diagram.png)

**Two System Options:**
- **Multi-Agent System**: 4 specialized agents (technical, sentiment, risk, portfolio) working collaboratively
- **Single-Agent System**: Unified adaptive agent with intelligent fallback protection

## Key Features

- **Multi-Agent Collaboration**: Four specialized agents working together for comprehensive analysis
- **Advanced Technical Analysis**: 20+ indicators including RSI, MACD, Bollinger Bands, volume analysis
- **Market Sentiment Analysis**: Real-time news sentiment and analyst recommendation analysis
- **Risk Management**: Comprehensive risk assessment with scenario modeling and position sizing
- **Smart Caching**: Intelligent data caching to minimize API calls and improve performance
- **Multiple Trading Strategies**: Support for swing trading, day trading, value investing, and momentum strategies

## Agent Specializations

### Market Data Analyst 🔍
- **Role**: Senior Technical Analyst
- **Expertise**: Institutional-grade technical analysis with advanced pattern recognition
- **Focus**: Multi-timeframe analysis, support/resistance levels, market microstructure
- **Tools**: Enhanced financial data analysis, Yahoo Finance integration

### Sentiment Analyst 📰
- **Role**: Senior Market Research Analyst  
- **Expertise**: Market psychology, fundamental research, macroeconomic factors
- **Focus**: News sentiment, analyst ratings, catalyst identification
- **Tools**: Web search, sentiment analysis, economic data integration

### Risk Manager ⚖️
- **Role**: Senior Risk Management Director
- **Expertise**: Quantitative modeling, stress testing, portfolio optimization
- **Focus**: Volatility modeling, position sizing, scenario analysis
- **Tools**: Portfolio analysis, risk metrics calculation, correlation analysis

### Portfolio Manager 💼
- **Role**: Chief Investment Officer
- **Expertise**: Investment strategy integration, capital allocation decisions
- **Focus**: Final investment decisions, portfolio optimization, execution strategy
- **Tools**: Comprehensive analysis integration, strategic decision making

## Quick Start

1. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run analysis**:
   ```bash
   python main.py
   ```

## Requirements

- Claude API key ([console.anthropic.com](https://console.anthropic.com))
- Serper API key ([serper.dev](https://serper.dev) - free tier available)
- Python 3.8+

## Trading Strategies Supported

- **Swing Trading**: Medium-term trends, 1-6 months holding period
- **Day Trading**: Short-term momentum, intraday to 1 week
- **Value Investing**: Fundamental value focus, 6 months to 2 years
- **Momentum Trading**: Trend continuation, 2-12 weeks
- **Custom Strategies**: User-defined parameters and focus areas

## Example Analysis Outputs

See real analysis examples demonstrating both system architectures:

### Multi-Agent Analysis
- [Amazon (AMZN) Comprehensive Analysis](analysis_examples/multi-agent-analysis/AMZN_Analysis.md) - Full 4-agent collaborative analysis

### Single-Agent Analysis  
- [Tesla (TSLA) Quick Analysis](analysis_examples/single-agent-analysis/TSLA_Analysis.md) - Rapid single-agent screening

*These examples showcase the depth and quality of analysis provided by each system architecture.*

## System Selection

Choose between architectures based on your needs:
- **Multi-agent**: Comprehensive analysis, higher quality, ~15 minutes
- **Single-agent**: Quick analysis, efficient execution, ~3 minutes

## Rate Limiting & Reliability

The system includes professional-grade rate limiting:
- Intelligent delays between API calls
- Context-aware processing delays  
- Smart caching to minimize API usage
- Automatic retry with exponential backoff
- Multiple fallback mechanisms

## Cost Estimation

- Multi-agent analysis: ~$0.15-0.25 per stock
- Single-agent analysis: ~$0.05-0.12 per stock

## Disclaimer

Educational and research purposes only. Not financial advice. Consult qualified financial advisors before making investment decisions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.