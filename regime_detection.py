import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import ruptures as rpt

def hmm_regime_detection(data, n_states=3):
    """
    Detects market regimes using a Hidden Markov Model (HMM).
    """
    # Fit HMM to price data
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(data[['price']].values)
    
    hidden_states = model.predict(data[['price']].values)
    return hidden_states

def change_point_detection(data):
    """
    Detects regime changes using change point detection algorithms.
    """
    algo = rpt.Pelt(model="rbf").fit(data['price'].values)
    result = algo.predict(pen=10)
    return result

if __name__ == "__main__":
    # Load the market data
    data = pd.read_csv('market_data.csv')
    
    # HMM for regime detection
    hidden_states = hmm_regime_detection(data)
    data['hmm_regime'] = hidden_states
    
    # Change point detection
    change_points = change_point_detection(data)
    data['change_points'] = 0
    data.loc[change_points, 'change_points'] = 1
    
    # Save the updated data with regimes
    data.to_csv('detected_regimes.csv', index=False)
    print("Regimes detected and saved to 'detected_regimes.csv'.")
