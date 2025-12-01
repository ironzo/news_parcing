# Economic News Scraping & RL Trading Bot 🚀

A sophisticated Python application that scrapes high-volatility economic news from Investing.com and uses Reinforcement Learning to generate trading signals.

## 🎯 Features

- **News Scraping**: Automated scraping of economic calendar from Investing.com
- **Smart Filtering**: Focuses on high-volatility USD news events
- **Database Storage**: SQLite/PostgreSQL support with SQLAlchemy ORM
- **RL Trading Agent**: Reinforcement Learning-based trading signal generation (Coming soon)
- **Market Data Integration**: Historical market data fetching via yfinance
- **Backtesting Framework**: Test trading strategies on historical data (Coming soon)
- **REST API**: FastAPI-based API for data access (Coming soon)
- **Docker Support**: Containerized deployment (Coming soon)

## 📁 Project Structure

```
news_parsing/
├── src/
│   ├── scrapers/          # Web scraping modules
│   │   └── investing_scraper.py
│   ├── database/          # Database models and repository
│   │   ├── models.py
│   │   └── repository.py
│   ├── utils/             # Utilities and configuration
│   │   ├── config.py
│   │   └── logger.py
│   └── analysis/          # RL and analysis modules (WIP)
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks for exploration
├── docker/                # Docker configuration (Coming soon)
├── main.py               # Main CLI entry point
├── requirements.txt      # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip or poetry

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/news_parsing.git
cd news_parsing
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (optional):
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Usage

#### Initialize Database

```bash
python main.py init-db
```

#### Scrape News

```bash
# Scrape and save to database
python main.py scrape

# Dry run (don't save to database)
python main.py scrape --dry-run
```

#### List Stored News

```bash
# List last 10 news items
python main.py list

# List last 20 items
python main.py list --limit 20
```

#### Show Configuration

```bash
python main.py config
```

#### Set Log Level

```bash
python main.py --log-level DEBUG scrape
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_scraper.py
```

## 🔧 Configuration

Configuration can be set via environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///news.db` |
| `INITIAL_CAPITAL` | Starting capital for trading | `10000.0` |
| `DEFAULT_SYMBOL` | Default trading symbol | `SPY` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_POSITION_SIZE` | Max position size (fraction) | `0.3` |

See `.env.example` for full list.

## 🤖 RL Trading Agent (Coming Soon)

The RL trading agent will:
- Use economic news data as features
- Learn optimal trading strategies via reinforcement learning
- Generate BUY/SELL/HOLD signals
- Support backtesting on historical data
- Provide real-time trading signals

### Planned Architecture

- **State**: Market conditions, news features, portfolio state
- **Actions**: BUY, SELL, HOLD
- **Reward**: Portfolio returns with risk adjustment
- **Algorithm**: PPO (Proximal Policy Optimization) via Stable-Baselines3

## 📊 Database Schema

### Tables

- **economic_news**: Economic calendar events
- **market_data**: OHLCV market data
- **trading_signals**: Generated trading signals
- **backtest_results**: Backtesting performance metrics

## 🐋 Docker Support (Coming Soon)

```bash
# Build and run
docker-compose up -d

# Run scraper
docker-compose exec app python main.py scrape
```

## 🛠️ Development

### Code Style

This project uses:
- **black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run linter
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement changes with tests
3. Run tests: `pytest`
4. Format code: `black .`
5. Commit and push
6. Create pull request

## 📝 Roadmap

### Phase 1: Foundation ✅ (Current)
- [x] Project restructuring
- [x] Web scraping module
- [x] Database models
- [x] CLI interface
- [x] Configuration management

### Phase 2: RL Implementation 🔄
- [ ] Market data fetching
- [ ] Feature engineering
- [ ] RL environment setup
- [ ] PPO agent training
- [ ] Signal generation

### Phase 3: Trading & Backtesting 📅
- [ ] Backtesting framework
- [ ] Performance metrics
- [ ] Visualization dashboard
- [ ] Paper trading mode

### Phase 4: Production 📅
- [ ] REST API with FastAPI
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Monitoring & alerts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This project is for educational purposes only. Trading financial instruments carries risk. Past performance does not guarantee future results. Always do your own research and consult with financial advisors before making trading decisions.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with** 🐍 Python, 🧠 Stable-Baselines3, 📊 SQLAlchemy, and ☕ caffeine
