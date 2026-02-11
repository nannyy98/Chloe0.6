# Chloe AI - Crypto & Stock Market Analysis Agent

An AI-powered agent for analyzing cryptocurrency and stock markets, identifying trading signals, and providing recommendations with risk assessment.

## 🎯 Purpose

Chloe AI analyzes market data to:
- Identify trading signals (Buy/Hold/Sell)
- Assess risk levels
- Provide explanations for decisions
- Learn from historical data
- Explain decisions in human language

⚠️ **Warning**: This is for educational/paper trading purposes only. Do not use for real trading without proper risk management.

## 🧱 Architecture

```
┌──────────────┐
│ Data Agents  │  ← bиржи, акции, новости
└──────┬───────┘
       ↓
┌──────────────┐
│ Feature Lab  │  ← индикаторы, признаки
└──────┬───────┘
       ↓
┌──────────────┐
│ ML Core      │  ← сигналы, вероятности
└──────┬───────┘
       ↓
┌──────────────┐
│ Risk Engine  │  ← защита депозита
└──────┬───────┘
       ↓
┌──────────────┐
│ LLM (Chloe)  │  ← объясняет, предупреждает
└──────────────┘
```

## 🧩 Tech Stack

- **Language**: Python 3.11+
- **Data**: ccxt, yfinance, pandas, numpy, ta
- **ML**: scikit-learn, xgboost, pytorch
- **LLM**: OpenAI/Ollama, langchain, crewAI
- **API/UI**: FastAPI, Telegram Bot

## 📅 Implementation Phases

### Phase 0 - Setup (Completed)
- Repository structure
- Dependencies

### Phase 1 - Data Collection (In Progress)
- OHLCV data collection for BTC, ETH
- Historical data storage

### Phase 2 - Indicators
- RSI, MACD, EMA (20/50/200)
- Volume and volatility indicators

### Phase 3 - ML Signals
- XGBoost/RandomForest models
- Buy/Hold/Sell predictions with probability

### Phase 4 - Backtesting
- Profit/loss analysis
- Max drawdown, win rate evaluation

### Phase 5 - Risk Engine
- Stop-loss, take-profit
- Position sizing controls

### Phase 6 - LLM Integration
- Signal explanation
- Risk warnings

### Phase 7 - Advanced Agents
- Market Agent, News Agent
- Strategy Agent, Risk Agent

### Phase 8 - Interface
- CLI, Telegram Bot, Web Dashboard

## 🧠 Survival Rules

1. ❌ Never trade real money
2. ❌ Don't trust "perfect signals"
3. ✅ Paper trading only
4. ✅ Logs + statistics
5. ✅ Emotion control# Chloe
# Chloe
# Chloe
