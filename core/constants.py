# ALL constants (risk-free rate, vol surfaces, etc.)

# Baseline Market Rates
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_VOLATILITY = 0.2

# Market Regime Boundaries (VIX based)
REGIME_LOW_VOL = 15.0
REGIME_HIGH_VOL = 30.0
REGIME_CRASH = 45.0

# Crisis Benchmarks (Ray Dalio & Peter Thiel logic)
DOT_COM_PEAK = "2000-03-10"
GFC_PEAK = "2007-10-09"
AI_BUBBLE_START = "2023-01-01"

# Inversion Principle Flags
FLAW_LACK_OF_JUMP = "Lack of Jump Diffusion"
FLAW_STATIC_VOL = "Constant Volatility Assumption"
FLAW_LACK_OF_FAT_TAILS = "Geometric Brownian Motion Limitation"

# CUDA Settings
USE_GPU = True
THREADS_PER_BLOCK = 256
