import pandas as pd
import matplotlib.pyplot as plt
from market_simulation import simulate_market_data
from regime_detection import hmm_regime_detection, change_point_detection
from adaptive_trading import AdaptiveTradingStrategy

if __name__ == "__main__":
    # Step 1: Simulate market data
    market_data = simulate_market_data()
    
    # Step 2: Detect market regimes
    market_data['hmm_regime'] = hmm_regime_detection(market_data)
    market_data['change_points'] = change_point_detection(market_data)
    
    # Step 3: Adaptive trading based on detected regimes
    strategy = AdaptiveTradingStrategy(initial_cash=10000)
    for i in range(1, len(market_data)):
        strategy.execute_trade(
            price=market_data['price'].iloc[i],
            regime=market_data['hmm_regime'].iloc[i],
            previous_price=market_data['price'].iloc[i-1]
        )
    
    print(f"Final Cash: {strategy.cash}")
    print(f"Final Position: {strategy.position}")

    # Step 4: Plot the price path with detected regimes
    plt.plot(market_data['price'], label="Price")
    plt.scatter(market_data.index, market_data['change_points'] * market_data['price'], color='red', label="Change Points")
    plt.title("Market Price with Detected Regimes")
    plt.legend()
    plt.show()
