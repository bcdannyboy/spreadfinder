# SpreadFinder

SpreadFinder is a Python tool designed to identify and rank optimal options spreads based on various criteria such as probability of success, return on risk (ROR), and options pricing. The tool supports both bull put spreads and iron condor spreads for specified stock symbols within a given date range.

## Features

- Fetches option chains for specified stock symbols within a given date range.
- Identifies and ranks the following spreads:
  - Bull Put
  - Iron Condor
- Utilizes Monte Carlo simulations to estimate probabilities of profit.
  - Implements Latin Hypercube sampling for more efficient simulations.
- Employs Black-Scholes option pricing.
- Implements Binomial pricing for American options.
- Uses GARCH model for estimating historical volatility.
  - Estimates volatility with / without taking the searched options DTE into account.
- Provides the Heston model for stochastic volatility simulations.
- Supports Student's t-distribution in Monte Carlo simulations for better tail risk modeling.
- Outputs the top spreads to a CSV file.
- Includes very basic visualization of the probabilities.
- Performs Bayesian Network analysis for better spread selection.
- **New:** Optional backtesting feature to evaluate the performance of identified spreads over historical data.

## Requirements

- Python 3.6+
- [Tradier](https://tradier.com/) API key for Options Chains
- [Financial Modeling Prep](https://site.financialmodelingprep.com/) API Key for Commodities
- `argparse`
- `datetime`
- `requests`
- `numpy`
- `pandas`
- `scipy`
- `arch`
- `ratelimit`
- `pgmpy`
- `matplotlib`
- `yfinance`

## Installation

Clone the repository:

```sh
git clone https://github.com/bcdannyboy/spreadfinder.git
```

Change to the project directory:

```sh
cd spreadfinder
```

Install the required Python packages:

```sh
pip install -r requirements.txt
```

## Usage

```sh
python spreadfinder.py -symbols SYMBOL,SYMBOL -mindte MINDTE -maxdte MAXDTE [-top_n TOP_N] [-min_ror MIN_ROR] [-max_strike_dist MAX_STRIKE_DIST] [-output OUTPUT] [-batch_size BATCH_SIZE] -api_token API_TOKEN [--include_iron_condors] [-simulations SIMULATIONS] [-risk_free_rate RISK_FREE_RATE] [--plot] [--backtesting] [-min_prob_success MIN_PROB_SUCCESS] [-commodities_api_key COMMODITIES_API_KEY]
```

### Arguments

- `-symbols`, `-s`: Comma-separated list of stock symbols to fetch data for (required).
- `-mindte`, `-m`: Minimum days to expiration (required).
- `-maxdte`, `-l`: Maximum days to expiration (required).
- `-top_n`, `-n`: Number of top spreads to display (default: 10).
- `-min_ror`, `-r`: Minimum return on risk (default: 0.15).
- `-max_strike_dist`, `-d`: Maximum strike distance percentage (default: 0.2).
- `-output`, `-o`: Output CSV file path (default: best_spreads.csv).
- `-batch_size`, `-b`: Batch size for processing tasks (default: 10).
- `-api_token`: Tradier API token (required).
- `--include_iron_condors`: Include iron condor spreads (default: False).
- `-simulations`, `-sim`: Number of Monte Carlo simulations to run (default: 1000).
- `-risk_free_rate`, `-rf`: Risk-free rate for option pricing (default: 0.01).
- `--plot`: Plot the probabilities (default: False).
- `--backtesting`: Enable backtesting (default: False).
- `-min_prob_success`, `-ps`: Minimum probability of success based on Bayesian probability (default: 0.5).
- `-commodities_api_key`, `-c`: FinancialModelingPrep API key for commodity data (required).

### Example

```sh
python spreadfinder.py -symbols GTLB -mindte 14 -maxdte 30 -top_n 10 -min_ror 0.15 -max_strike_dist 0.5 -batch_size 10000 -api_token "2FRbqCT74L2iJBpAFArYZTThhNxU" -simulations 1000 -risk_free_rate 0.0454 -commodities_api_key CMy2CnXAdirIafquonQ0n9jKNZmKDruP --plot
```

This command fetches option chains for GTLB with expirations between 14 and 30 days, calculates historical volatility using the GARCH model, finds the top 10 spreads with a minimum return on risk of 15% and a maximum strike distance of 50%, runs 1000 Monte Carlo simulations for each spread using Student's t-distribution, and plots the results.

### Additional Notes

#### GARCH Model
The Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model is used for estimating the volatility of financial returns. It models the variance of the current error term as a function of the variances of previous time periods, providing a more accurate estimate of volatility over time.

#### Student's t-Distribution
The Student's t-distribution is used in Monte Carlo simulations to model the heavier tails observed in financial return distributions, capturing extreme events more effectively than the normal distribution.

#### Heston Model
The Heston model is a stochastic volatility model that assumes volatility is not constant but follows its own random process. It is widely used for pricing derivatives and capturing the empirical features of market data, such as volatility clustering and the leverage effect.

#### Bayesian Network Analysis
Bayesian Network analysis is used to model the relationships between different variables in the option spread selection process. It helps identify the most relevant factors that influence the success of a spread and provides a probabilistic framework for decision-making.

### Notes

- Iron Condors take **super** long right now.
- Ensure you have a valid Tradier API token to fetch real-time options data.
- Tradier API requests are rate-limited to 60/minute.
- When calculating volatility, the Monte Carlo simulation uses the GARCH model both with and without taking into account the DTE of the options you're searching for.
- When selling options, you typically want to identify spreads that are fairly priced or overpriced to maximize your return on risk.

### TODO

- [ ] Better visualizations
- [ ] More spreads
- [ ] Better Backtesting