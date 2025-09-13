# AI Financial Trading & Analysis System

Financial trading analysis system with dual AI architectures: multi-agent collaborative approach and single-agent adaptive approach, both powered by Claude AI.

## Architecture

![Multi-Agent Architecture](multi-agent_architecture_diagram.png)

**Two System Options:**
- **Multi-Agent System**: 4 specialized agents (technical, sentiment, risk, portfolio) working collaboratively
- **Single-Agent System**: Unified adaptive agent with intelligent fallback protection

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

## Features

- Technical analysis with 20+ indicators
- Market sentiment analysis from news and analyst ratings
- Risk management with position sizing recommendations
- Portfolio optimization and correlation analysis
- Smart caching system to minimize API costs
- Professional rate limiting and fallback protection
- Multiple trading strategies (swing, day trading, value, momentum)

## System Selection

Choose between architectures based on your needs:
- **Multi-agent**: Comprehensive analysis, higher quality, ~15 minutes
- **Single-agent**: Quick analysis, efficient execution, ~3 minutes

## Cost Estimation

- Multi-agent analysis: ~$0.15-0.25 per stock
- Single-agent analysis: ~$0.05-0.12 per stock

## Disclaimer

Educational and research purposes only. Not financial advice. Consult qualified financial advisors before making investment decisions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.