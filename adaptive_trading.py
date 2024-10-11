import numpy as np
import pandas as pd

class AdaptiveTradingStrategy:
    def __init__(self, initial_cash=10000):
        self.cash = initial_cash
        self.position = 0

    def momentum_strategy(self, price_change):
        """Trading strategy for trending regime: go with the trend."""
        if price_change > 0:
            self.position += 1  # Buy
        elif price_change < 0:
            self.position -= 1  # Sell

    def mean_reverting_strategy(self, price_change):
        """Trading strategy for mean-reverting regime: bet against the trend."""
        if price_change > 0:
            self.position -= 1  # Sell
        elif price_change < 0:
            self.position += 1  # Buy

    def volatile_strategy(self, price_change):
        """Trading strategy for volatile regime: reduce risk by scaling down."""
        self.position = self.position * 0.5  # Reduce position size in volatile market

    def execute_trade(self, price, regime, previous_price):
        price_change = price - previous_price
        
        # Select trading strategy based on the detected regime
        if regime == 'trending':
            self.momentum_strategy(price_change)
        elif regime == 'mean-reverting':
            self.mean_reverting_strategy(price_change)
        elif regime == 'volatile':
            self.volatile_strategy(price_change)

        # Update cash balance based on position
        self.cash += self.position * price_change

if __name__ == "__main__":
    # Load the data with detected regimes
    data = pd.read_csv('detected_regimes.csv')
    
    # Initialize the trading strategy
    strategy = AdaptiveTradingStrategy(initial_cash=10000)
    
    # Simulate the adaptive trading
    for i in range(1, len(data)):
        strategy.execute_trade(
            price=data['price'].iloc[i],
            regime=data['regime'].iloc[i],
            previous_price=data['price'].iloc[i-1]
        )

    # Output final cash and position
    print(f"Final Cash: {strategy.cash}")
    print(f"Final Position: {strategy.position}")
