from enhanced_risk_engine import get_enhanced_risk_engine

engine = get_enhanced_risk_engine()
print('Enhanced Risk Engine test:')
print(f'Initial capital: ${engine.initial_capital:,.2f}')

# Test Kelly sizing
kelly_size = engine.calculate_kelly_position_size(0.6, 2.0, 100000, 'STABLE')
print(f'Kelly position size: ${kelly_size:,.2f}')

# Test risk assessment
risk_assessment = engine.assess_position_risk(
    symbol='BTC/USDT',
    entry_price=50000,
    position_size=0.1,
    stop_loss=48000,
    take_profit=55000,
    volatility=0.03,
    regime='STABLE'
)
print(f'Risk assessment approved: {risk_assessment.approved}')
if risk_assessment.rejection_reason:
    print(f'Rejection reason: {risk_assessment.rejection_reason}')
else:
    print('Position approved!')
    print(f'Risk/Reward ratio: {risk_assessment.risk_metrics["risk_reward_ratio"]:.2f}')
    print(f'Position size: {risk_assessment.risk_metrics["position_percentage"]:.1f}%')