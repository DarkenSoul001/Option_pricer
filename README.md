# Option Pricer Project

A comprehensive options pricing engine featuring multiple models (Black-Scholes, Binomial, Heston, etc.), ML-based regime classification, and crisis stress testing.

## architecture
- **core/**: Shared constants, math engine, and market regime classifiers.
- **models/**: Implementation of various option pricing models.
- **ml/**: Machine learning components for volatility forecasting and model selection.
- **crisis_data/**: Historical crisis data and loaders.
- **stress_test/**: Scenario analysis for various market bubbles.
- **analysis/**: Performance comparison and visualization tools.
- **dashboard.py**: Streamlit-based user interface.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```
