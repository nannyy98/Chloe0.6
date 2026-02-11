"""
Web Interface for Chloe AI 0.4
Professional trading dashboard with real-time monitoring and controls
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logger = logging.getLogger(__name__)

@dataclass
class DashboardData:
    """Dashboard data structure"""
    timestamp: str
    portfolio_value: float
    portfolio_return: float
    current_drawdown: float
    active_positions: int
    regime_state: str
    regime_confidence: float
    risk_metrics: Dict
    market_sentiment: float
    system_status: str
    alerts: List[Dict]
    performance_chart: List[Dict]
    positions_data: List[Dict]

class WebSocketManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if self.active_connections:
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")
                    self.disconnect(connection)

class WebInterface:
    """Main web interface controller"""
    
    def __init__(self):
        self.websocket_manager = WebSocketManager()
        self.dashboard_data = DashboardData(
            timestamp=datetime.now().isoformat(),
            portfolio_value=100000.0,
            portfolio_return=0.0,
            current_drawdown=0.0,
            active_positions=0,
            regime_state="STABLE",
            regime_confidence=0.8,
            risk_metrics={},
            market_sentiment=0.0,
            system_status="RUNNING",
            alerts=[],
            performance_chart=[],
            positions_data=[]
        )
        logger.info("Web Interface initialized")

    async def update_dashboard_data(self, new_data: DashboardData):
        """Update dashboard data and broadcast to clients"""
        self.dashboard_data = new_data
        await self.websocket_manager.broadcast({
            "type": "dashboard_update",
            "data": self._serialize_dashboard_data(new_data)
        })

    def _serialize_dashboard_data(self, data: DashboardData) -> dict:
        """Convert DashboardData to serializable dict"""
        return {
            "timestamp": data.timestamp,
            "portfolio_value": data.portfolio_value,
            "portfolio_return": data.portfolio_return,
            "current_drawdown": data.current_drawdown,
            "active_positions": data.active_positions,
            "regime_state": data.regime_state,
            "regime_confidence": data.regime_confidence,
            "risk_metrics": data.risk_metrics,
            "market_sentiment": data.market_sentiment,
            "system_status": data.system_status,
            "alerts": data.alerts,
            "performance_chart": data.performance_chart,
            "positions_data": data.positions_data
        }

# Initialize FastAPI app
app = FastAPI(title="Chloe AI 0.4 Dashboard", version="1.0.0")

# Initialize web interface
web_interface = WebInterface()

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    logger.warning("Static directory not found, creating basic interface")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve main dashboard page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chloe AI 0.4 - Professional Trading Dashboard</title>
        <style>
            :root {
                --primary-color: #2563eb;
                --secondary-color: #1e40af;
                --success-color: #10b981;
                --warning-color: #f59e0b;
                --danger-color: #ef4444;
                --dark-bg: #0f172a;
                --card-bg: #1e293b;
                --text-primary: #f1f5f9;
                --text-secondary: #94a3b8;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, var(--dark-bg) 0%, #1e293b 100%);
                color: var(--text-primary);
                min-height: 100vh;
            }
            
            .dashboard-container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 30px;
                padding: 20px;
                background: var(--card-bg);
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .logo {
                display: flex;
                align-items: center;
                gap: 15px;
            }
            
            .logo-icon {
                width: 40px;
                height: 40px;
                background: var(--primary-color);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 20px;
            }
            
            .status-indicator {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px 16px;
                background: var(--success-color);
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
            }
            
            .status-indicator.offline {
                background: var(--danger-color);
            }
            
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .card {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            
            .card-title {
                font-size: 18px;
                font-weight: 600;
                color: var(--text-primary);
            }
            
            .metric-value {
                font-size: 28px;
                font-weight: 700;
                margin: 10px 0;
            }
            
            .metric-positive {
                color: var(--success-color);
            }
            
            .metric-negative {
                color: var(--danger-color);
            }
            
            .metric-neutral {
                color: var(--text-secondary);
            }
            
            .progress-bar {
                height: 8px;
                background: #334155;
                border-radius: 4px;
                overflow: hidden;
                margin: 15px 0;
            }
            
            .progress-fill {
                height: 100%;
                background: var(--primary-color);
                border-radius: 4px;
                transition: width 0.3s ease;
            }
            
            .alert-list {
                max-height: 200px;
                overflow-y: auto;
            }
            
            .alert-item {
                padding: 12px;
                margin-bottom: 10px;
                border-radius: 8px;
                border-left: 4px solid;
            }
            
            .alert-warning {
                background: rgba(245, 158, 11, 0.1);
                border-left-color: var(--warning-color);
            }
            
            .alert-info {
                background: rgba(37, 99, 235, 0.1);
                border-left-color: var(--primary-color);
            }
            
            .positions-table {
                width: 100%;
                border-collapse: collapse;
            }
            
            .positions-table th,
            .positions-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #334155;
            }
            
            .positions-table th {
                color: var(--text-secondary);
                font-weight: 600;
            }
            
            .chart-container {
                height: 300px;
                background: #0f172a;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: var(--text-secondary);
            }
            
            .connection-status {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 20px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
                z-index: 1000;
            }
            
            .connected {
                background: var(--success-color);
            }
            
            .disconnected {
                background: var(--danger-color);
            }
            
            @media (max-width: 768px) {
                .dashboard-container {
                    padding: 15px;
                }
                
                .grid {
                    grid-template-columns: 1fr;
                }
                
                .header {
                    flex-direction: column;
                    gap: 15px;
                    text-align: center;
                }
            }
        </style>
    </head>
    <body>
        <div id="connectionStatus" class="connection-status disconnected">Disconnected</div>
        
        <div class="dashboard-container">
            <div class="header">
                <div class="logo">
                    <div class="logo-icon">AI</div>
                    <div>
                        <h1>Chloe AI 0.4</h1>
                        <p>Professional Trading Dashboard</p>
                    </div>
                </div>
                <div id="systemStatus" class="status-indicator">
                    <span>●</span>
                    <span>RUNNING</span>
                </div>
            </div>
            
            <div class="grid">
                <!-- Portfolio Overview -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Portfolio Overview</div>
                        <div id="portfolioReturn" class="metric-neutral">0.00%</div>
                    </div>
                    <div class="metric-value" id="portfolioValue">$100,000.00</div>
                    <div>Current Value</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="drawdownProgress" style="width: 0%"></div>
                    </div>
                    <div>Drawdown: <span id="currentDrawdown">0.00%</span></div>
                </div>
                
                <!-- Market Regime -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Market Regime</div>
                        <div id="regimeConfidence" class="metric-neutral">80%</div>
                    </div>
                    <div class="metric-value" id="regimeState">STABLE</div>
                    <div>Current Market State</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="regimeProgress" style="width: 80%"></div>
                    </div>
                    <div>Risk Adjustment Applied</div>
                </div>
                
                <!-- Risk Metrics -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Risk Metrics</div>
                        <div id="riskLevel" class="metric-neutral">MODERATE</div>
                    </div>
                    <div>VaR (95%): <span id="varMetric">2.00%</span></div>
                    <div>Max Drawdown: <span id="maxDrawdown">5.00%</span></div>
                    <div>Sharpe Ratio: <span id="sharpeRatio">1.20</span></div>
                    <div>Correlation Risk: <span id="correlationRisk">0.30</span></div>
                </div>
                
                <!-- Market Sentiment -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Market Sentiment</div>
                        <div id="sentimentIndicator" class="metric-neutral">NEUTRAL</div>
                    </div>
                    <div class="metric-value" id="marketSentiment">0.00</div>
                    <div>News & Social Analysis</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="sentimentProgress" style="width: 50%"></div>
                    </div>
                    <div>Bullish/Bearish Indicator</div>
                </div>
            </div>
            
            <div class="grid">
                <!-- Active Positions -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Active Positions</div>
                        <div id="activePositionsCount" class="metric-neutral">0</div>
                    </div>
                    <table class="positions-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Size</th>
                                <th>P&L</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="positionsTableBody">
                            <tr>
                                <td colspan="4" style="text-align: center; color: var(--text-secondary);">
                                    No active positions
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <!-- System Alerts -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">System Alerts</div>
                        <div id="alertsCount" class="metric-neutral">0</div>
                    </div>
                    <div class="alert-list" id="alertsList">
                        <div class="alert-item alert-info">
                            System initialized and running normally
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Performance Chart -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Performance Chart</div>
                    <div>Last 30 Days</div>
                </div>
                <div class="chart-container" id="performanceChart">
                    Real-time performance chart will appear here
                </div>
            </div>
        </div>
        
        <script>
            let socket = null;
            let isConnected = false;
            
            function connectWebSocket() {
                const wsUrl = `ws://${window.location.host}/ws`;
                socket = new WebSocket(wsUrl);
                
                socket.onopen = function(event) {
                    console.log('WebSocket connected');
                    isConnected = true;
                    updateConnectionStatus(true);
                };
                
                socket.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'dashboard_update') {
                        updateDashboard(data.data);
                    }
                };
                
                socket.onclose = function(event) {
                    console.log('WebSocket disconnected');
                    isConnected = false;
                    updateConnectionStatus(false);
                    // Attempt to reconnect
                    setTimeout(connectWebSocket, 3000);
                };
                
                socket.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }
            
            function updateConnectionStatus(connected) {
                const statusElement = document.getElementById('connectionStatus');
                if (connected) {
                    statusElement.className = 'connection-status connected';
                    statusElement.textContent = 'Connected';
                } else {
                    statusElement.className = 'connection-status disconnected';
                    statusElement.textContent = 'Disconnected';
                }
            }
            
            function updateDashboard(data) {
                // Update portfolio metrics
                document.getElementById('portfolioValue').textContent = 
                    `$${data.portfolio_value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
                
                const returnElement = document.getElementById('portfolioReturn');
                const returnClass = data.portfolio_return >= 0 ? 'metric-positive' : 'metric-negative';
                returnElement.className = returnClass;
                returnElement.textContent = `${data.portfolio_return >= 0 ? '+' : ''}${data.portfolio_return.toFixed(2)}%`;
                
                document.getElementById('currentDrawdown').textContent = `${data.current_drawdown.toFixed(2)}%`;
                document.getElementById('drawdownProgress').style.width = `${Math.min(100, data.current_drawdown * 10)}%`;
                
                // Update regime
                document.getElementById('regimeState').textContent = data.regime_state;
                document.getElementById('regimeConfidence').textContent = `${(data.regime_confidence * 100).toFixed(0)}%`;
                document.getElementById('regimeProgress').style.width = `${data.regime_confidence * 100}%`;
                
                // Update risk metrics
                if (data.risk_metrics) {
                    document.getElementById('varMetric').textContent = `${(data.risk_metrics.var_95 * 100 || 2).toFixed(2)}%`;
                    document.getElementById('maxDrawdown').textContent = `${(data.risk_metrics.max_drawdown * 100 || 5).toFixed(2)}%`;
                    document.getElementById('sharpeRatio').textContent = (data.risk_metrics.sharpe_ratio || 1.2).toFixed(2);
                    document.getElementById('correlationRisk').textContent = (data.risk_metrics.correlation_risk || 0.3).toFixed(2);
                }
                
                // Update sentiment
                document.getElementById('marketSentiment').textContent = data.market_sentiment.toFixed(2);
                const sentimentText = data.market_sentiment > 0.2 ? 'BULLISH' : 
                                    data.market_sentiment < -0.2 ? 'BEARISH' : 'NEUTRAL';
                document.getElementById('sentimentIndicator').textContent = sentimentText;
                document.getElementById('sentimentProgress').style.width = `${50 + (data.market_sentiment * 25)}%`;
                
                // Update positions
                document.getElementById('activePositionsCount').textContent = data.active_positions;
                
                if (data.positions_data && data.positions_data.length > 0) {
                    const tbody = document.getElementById('positionsTableBody');
                    tbody.innerHTML = data.positions_data.map(pos => `
                        <tr>
                            <td>${pos.symbol}</td>
                            <td>${pos.size.toFixed(4)}</td>
                            <td class="${pos.pnl >= 0 ? 'metric-positive' : 'metric-negative'}">
                                ${pos.pnl >= 0 ? '+' : ''}${pos.pnl.toFixed(2)}%
                            </td>
                            <td>${pos.status}</td>
                        </tr>
                    `).join('');
                }
                
                // Update alerts
                const alertsContainer = document.getElementById('alertsList');
                if (data.alerts && data.alerts.length > 0) {
                    alertsContainer.innerHTML = data.alerts.map(alert => `
                        <div class="alert-item alert-${alert.severity.toLowerCase()}">
                            ${alert.message}
                        </div>
                    `).join('');
                }
                
                // Update timestamp
                document.getElementById('systemStatus').innerHTML = 
                    `<span>●</span><span>${data.system_status}</span>`;
            }
            
            // Initialize when page loads
            window.onload = function() {
                connectWebSocket();
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await web_interface.websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        web_interface.websocket_manager.disconnect(websocket)

@app.get("/api/dashboard")
async def get_dashboard_data():
    """API endpoint for dashboard data"""
    return web_interface._serialize_dashboard_data(web_interface.dashboard_data)

@app.post("/api/update")
async def update_dashboard(update_data: dict):
    """API endpoint to update dashboard data"""
    try:
        new_data = DashboardData(**update_data)
        await web_interface.update_dashboard_data(new_data)
        return {"status": "success", "message": "Dashboard updated"}
    except Exception as e:
        logger.error(f"Failed to update dashboard: {e}")
        return {"status": "error", "message": str(e)}

def get_web_app():
    """Get the FastAPI application instance"""
    return app

def main():
    """Start the web server"""
    logger.info("Starting Chloe AI Web Interface...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()