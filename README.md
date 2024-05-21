# spreadfinder

spreadfinder is a Python tool that identifies and ranks optimal options spreads based on probability of success, return on risk (ROR), and options pricing. The tool supports finding bull put spreads and iron condor spreads for a given stock symbol within a specified date range.

## Features
Fetches option chains for a given stock symbol within a specified date range.

- Spreads Ranked:
    - Bull Put
    - Iron Condor
- Ranks spreads based on probability of success and return on risk.
- Outputs the top spreads to a CSV file.

## Requirements
- Python 3.6+
- `argparse`
- `datetime`
- `yfinance`
- `numpy`
- `pandas`
- `scipy`

## Installation
Clone the repository:

```sh
git clone https://github.com/yourusername/spreadfinder.git
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
python spreadfinder.py -symbol SYMBOL -mindte MINDTE -maxdte MAXDTE [-top_n TOP_N] [-min_ror MIN_ROR] [-max_strike_dist MAX_STRIKE_DIST] [-output OUTPUT]
Arguments
-symbol, -s: Stock symbol to fetch data for (required).
-mindte, -m: Minimum days to expiration (required).
-maxdte, -l: Maximum days to expiration (required).
-top_n, -n: Number of top spreads to display (default: 10).
-min_ror, -r: Minimum return on risk (default: 0.15).
-max_strike_dist, -d: Maximum strike distance percentage (default: 0.2).
-output, -o: Output CSV file path (default: best_spreads.csv).
-max_workers, -w: Maximum number of workers for parallel processing (default: 4).
```

### Example
```sh
python spreadfinder.py -symbol AAPL -mindte 30 -maxdte 60 -top_n 5 -min_ror 0.2 -max_strike_dist 0.15 -w 10 -output aapl_spreads.csv
```

This command fetches option chains for AAPL with expirations between 30 and 60 days, finds the top 5 spreads with a minimum return on risk of 20% and a maximum strike distance of 15%, and saves the results to aapl_spreads.csv.
