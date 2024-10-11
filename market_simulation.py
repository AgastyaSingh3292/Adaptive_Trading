import numpy as np
import pandas as pd

def simulate_market_data(num_steps=1000):
    """
    Simulates market data with three regimes: trending, mean-reverting, and volatile.
    """
    np.random.seed(42)
    price = 100
    price_path = [price]
    regime = []
    
    for step in range(num_steps):
        if step < 0.33 * num_steps:
            # Trending regime: upward drift
            price += np.random.normal(loc=0.2, scale=0.5)
            regime.append("trending")
        elif step < 0.66 * num_steps:
            # Mean-reverting regime: small oscillations around mean
            price += np.random.normal(loc=0.0, scale=0.2)
            regime.append("mean-reverting")
        else:
            # Volatile regime: large fluctuations
            price += np.random.normal(loc=0.0, scale=1.0)
            regime.append("volatile")
        price_path.append(price)
    
    data = pd.DataFrame({
        'price': price_path[:-1],
        'regime': regime
    })
    
    return data

if __name__ == "__main__":
    data = simulate_market_data()
    data.to_csv('market_data.csv', index=False)
    print("Market data generated and saved to 'market_data.csv'.")
