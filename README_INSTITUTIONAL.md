# Chloe Institutional Trading Platform

[![CI/CD](https://github.com/yourusername/chloe-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/chloe-ai/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🏛️ Institutional-Grade Algorithmic Trading Platform

Chloe is a professional algorithmic trading platform designed for institutional use, featuring event-driven architecture, comprehensive risk management, and production-ready infrastructure.

## 🚀 Key Features

### 🔧 Core Architecture
- **Event-Driven Design**: Professional event bus architecture like hedge funds
- **Portfolio-First Approach**: Risk management before signal generation
- **Modular Components**: Separation of concerns for maintainability
- **Real-time Processing**: Asynchronous execution with proper error handling

### 💰 Trading Infrastructure
- **Multi-Broker Support**: Binance, Interactive Brokers, and custom broker adapters
- **Professional Risk Management**: Institutional risk limits with circuit breaker
- **Advanced Position Sizing**: Volatility-adjusted position sizing algorithms
- **Real Market Conditions**: Slippage, commissions, and market impact modeling

### 📊 Data & Analysis
- **Professional Data Pipeline**: Parquet storage with data validation
- **Feature Engineering**: 50+ technical indicators and custom features
- **Lookahead Bias Prevention**: Built-in bias detection mechanisms
- **Multi-Asset Support**: Cryptocurrency, stocks, forex, and commodities

### 🛡️ Risk Management
- **Portfolio-Level Risk**: Exposure limits and concentration controls
- **Dynamic Drawdown Protection**: Automatic circuit breaker activation
- **Position-Level Risk**: Individual trade risk assessment
- **Real-time Monitoring**: Continuous risk metric calculation

## 🏗️ Architecture Overview

```
Event Bus (Core Engine)
    ↓
Portfolio Manager (Capital Allocation)
    ↓
Risk Engine (Risk Assessment)
    ↓
Order Manager (Execution)
    ↓
Data Pipeline (Market Feeds)
```

## 📦 Installation

### Prerequisites
- Python 3.9+
- Docker (optional but recommended)
- Redis (for production deployments)

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/chloe-ai.git
cd chloe-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run paper trading demo
python institutional_main.py --mode paper --symbols BTC/USDT ETH/USDT
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access services:
# - Trading Platform: http://localhost:8000
# - Grafana Dashboard: http://localhost:3000 (admin/admin123)
# - Prometheus Metrics: http://localhost:9090
```

## 🎯 Usage Examples

### 1. Paper Trading Mode
```python
from institutional_main import TradingOrchestrator

# Initialize trading system
orchestrator = TradingOrchestrator(
    initial_capital=100000.0,
    mode='paper'
)

# Start trading
await orchestrator.start_trading(['BTC/USDT', 'ETH/USDT'])
```

### 2. Backtesting
```bash
# Run comprehensive backtest
python institutional_main.py --mode backtest --symbols BTC/USDT --days 365

# Results include:
# - Sharpe Ratio
# - Maximum Drawdown
# - Win Rate
# - Profit Factor
# - Transaction Costs Analysis
```

### 3. Custom Strategy Development
```python
from core.event_bus import SignalEvent
from portfolio.portfolio import portfolio

async def my_strategy(symbol: str, market_data: dict):
    """Example custom strategy"""
    # Your trading logic here
    if should_buy(symbol, market_data):
        signal_event = SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=datetime.now(),
            data={'strategy': 'my_strategy'},
            symbol=symbol,
            signal='BUY',
            confidence=0.85,
            strategy_id='custom_strategy_1'
        )
        await event_bus.publish(signal_event)
```

## 📈 Performance Metrics

### Risk Management
- **Maximum Drawdown**: 15% default limit
- **Position Concentration**: 25% maximum per position
- **Daily Loss Limit**: 3% maximum daily loss
- **Portfolio Exposure**: 5% maximum total exposure

### Trading Performance
- **Slippage Modeling**: Realistic 0.05% average slippage
- **Commission Costs**: 0.1% trading fees
- **Position Sizing**: Volatility-adjusted risk management
- **Execution Quality**: Market impact modeling

## 🔧 Configuration

### Environment Variables
```bash
export TRADING_MODE=paper          # paper, live, backtest
export INITIAL_CAPITAL=100000      # Starting capital
export RISK_PER_TRADE=0.01         # 1% risk per trade
export MAX_DRAWDOWN=0.15           # 15% maximum drawdown
export LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
```

### Risk Limits Configuration
```python
risk_limits = RiskLimits(
    max_risk_per_trade=0.01,        # 1% per trade
    max_portfolio_risk=0.05,        # 5% total exposure  
    max_drawdown=0.15,              # 15% max drawdown
    max_daily_loss=0.03,            # 3% daily loss limit
    max_position_concentration=0.25 # 25% per position
)
```

## 🛡️ Security & Compliance

### Production Security
- **API Key Management**: Secure credential storage
- **Rate Limiting**: Exchange API protection
- **Audit Logging**: Comprehensive transaction logging
- **Data Encryption**: At-rest and in-transit encryption

### Risk Controls
- **Circuit Breaker**: Automatic trading suspension
- **Kill Switch**: Emergency position liquidation
- **Position Limits**: Hard position size constraints
- **Time-based Controls**: Trading hour restrictions

## 📊 Monitoring & Analytics

### Built-in Dashboards
- **Grafana Integration**: Real-time performance metrics
- **Prometheus Metrics**: System health monitoring
- **Custom Alerts**: Risk limit violation notifications
- **Performance Reports**: Automated daily/weekly reports

### Key Metrics Tracked
- Portfolio equity curve
- Risk-adjusted returns (Sharpe, Sortino)
- Drawdown analysis
- Trade execution quality
- Cost analysis (commissions, slippage)

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=chloe --cov-report=html
```

### Integration Tests
```bash
# Test trading workflows
pytest tests/integration/ -v

# Test risk management
pytest tests/risk/ -v
```

## 🚀 Production Deployment

### Kubernetes Deployment
```bash
# Deploy to Kubernetes cluster
kubectl apply -f kubernetes/

# Monitor deployment
kubectl get pods
kubectl logs -f deployment/chloe-trading
```

### Cloud Providers
- **AWS**: ECS/EKS deployment configurations
- **GCP**: GKE deployment templates
- **Azure**: AKS deployment guides

## 📚 Documentation

### API Documentation
- [Trading API](docs/api.md)
- [Risk Management](docs/risk.md)
- [Data Pipeline](docs/data.md)
- [Event System](docs/events.md)

### Developer Guides
- [Strategy Development](docs/strategies.md)
- [Custom Indicators](docs/indicators.md)
- [Broker Integration](docs/brokers.md)
- [Performance Optimization](docs/performance.md)

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Code formatting
black .
isort .

# Type checking
mypy .
```

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies and financial instruments involves substantial risk of loss. Past performance is not indicative of future results. Do not risk money you cannot afford to lose.

**Always test strategies thoroughly in paper trading before considering live trading.**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with inspiration from institutional trading systems
- Uses industry-standard libraries and best practices
- Designed for transparency and educational purposes

---

*"Professional trading requires professional tools."*