# SpreadFinder

SpreadFinder is a single-file Python utility for finding and ranking bull put credit spreads from live Tradier option-chain data. After setting a Tradier token, execution stays simple:

```sh
export TRADIER_API_TOKEN="your_tradier_token"
python spreadfinder.py --symbols AAPL --min-dte 14 --max-dte 45
```

The tool scores candidate spreads with probability, return on risk, expected value, pricing edge, and simple liquidity signals, then writes the ranked results to a CSV file.

## Current Features

- Fetches Tradier option expirations and option chains.
- Finds bull put credit spreads in a requested DTE range.
- Estimates volatility with GARCH when `arch` is installed, with a realized-volatility fallback.
- Runs Monte Carlo terminal-price simulations for profit and max-profit probabilities.
- Adds practical spread fields such as DTE, width, breakeven, per-contract credit, max loss, annualized ROR, expected value, and bid/ask spread.
- Supports optional Financial Modeling Prep commodity correlation scoring when enabled with a key.
- Includes a no-network smoke test with `--self-test`.

Iron condors, plotting, and backtesting are reserved but not implemented. Their legacy flags fail fast instead of crashing or silently doing nothing.

## Requirements

- Python 3.10+.
- A [Tradier](https://tradier.com/) API token for option chains and quote/history data. The script targets Tradier's production API at `https://api.tradier.com/v1`.
- Optional: a [Financial Modeling Prep](https://site.financialmodelingprep.com/) API key for commodity correlation scoring.

Install core dependencies:

```sh
python -m pip install -r requirements.txt
```

Optional quality/features:

```sh
python -m pip install scipy arch yfinance
```

`scipy` improves Monte Carlo sampling, `arch` enables GARCH volatility estimates, and `yfinance` is required only when commodity scoring is enabled.

## Quick Start

First verify the script and installed core dependencies without any network calls:

```sh
python spreadfinder.py --self-test
```

Set your Tradier token as an environment variable so it does not appear in shell history.

macOS/Linux:

```sh
export TRADIER_API_TOKEN="your_tradier_token"
python spreadfinder.py --symbols AAPL --min-dte 14 --max-dte 45
```

PowerShell:

```powershell
$env:TRADIER_API_TOKEN = "your_tradier_token"
python spreadfinder.py --symbols AAPL --min-dte 14 --max-dte 45
```

Optional commodity scoring:

```sh
export FMP_API_KEY="your_fmp_key"
python spreadfinder.py --symbols AAPL --min-dte 14 --max-dte 45 --use-commodity-score
```

Commodity scoring fetches a Financial Modeling Prep commodity list/history and Yahoo Finance stock history, capped by `--max-commodities`. It is opt-in because it is slower, uses more network calls, and needs optional `yfinance`.

## Usage

```sh
python spreadfinder.py --symbols SYMBOL[,SYMBOL...] --min-dte DAYS --max-dte DAYS [options]
```

Common options:

- `--symbols`, `-s`: comma-separated stock symbols. Symbols are stripped, uppercased, and deduplicated.
- `--min-dte`, `-m`: minimum days to expiration.
- `--max-dte`, `-l`: maximum days to expiration.
- `--top`, `-n`: number of top ranked spreads to display and write.
- `--min-ror`, `-r`: minimum return on risk as a decimal. `0.15` means 15%.
- `--max-strike-dist`, `-d`: maximum short-strike distance from spot as a decimal. `0.2` means 20%.
- `--min-prob-success`, `-ps`: minimum composite probability as a decimal.
- `--simulations`: Monte Carlo simulations per expiration.
- `--risk-free-rate`, `-rf`: risk-free rate as a decimal. `0.01` means 1%.
- `--output`, `-o`: output CSV path. Default: `best_spreads.csv`.
- `--batch-size`, `-b`: maximum concurrent option-chain requests, capped by local CPU and internal worker limits.
- `--api-token`: Tradier token. Prefer `TRADIER_API_TOKEN`.
- `--commodities-api-key`, `-c`: FMP key. Prefer `FMP_API_KEY`.
- `--use-commodity-score`: include optional FMP commodity correlation in the composite score.
- `--max-commodities`: maximum FMP commodity symbols sampled when commodity scoring is enabled.
- `--self-test`: run the built-in smoke test and exit.
- `--quiet`: suppress progress logs.
- `--debug`: show tracebacks for troubleshooting.
- `--version`: print the SpreadFinder version.

Legacy aliases such as `-symbols`, `-mindte`, `-maxdte`, `-top_n`, `-api_token`, and `-commodities_api_key` are still accepted for compatibility.

Examples:

```sh
python spreadfinder.py --symbols AAPL,MSFT --min-dte 21 --max-dte 60 --output tech_spreads.csv
python spreadfinder.py --symbols SPY --min-dte 14 --max-dte 45 --min-ror 0.20 --min-prob-success 0.60
python spreadfinder.py --symbols AAPL --min-dte 14 --max-dte 45 --quiet
```

## Output

SpreadFinder always writes a CSV, even when no spread matches. Key output columns include:

- legs/context: symbol, spread type, expiration, DTE, underlying price, volatility used, short put, long put, short-put distance from spot.
- risk: width, breakeven, credit, max profit, max loss, per-contract values.
- ranking: return on risk, annualized ROR, probability fields, expected value, composite score.
- quality: pricing state, bid/ask spreads, minimum volume, minimum open interest.

Credit and max loss are shown per share and per standard 100-share contract.

The default output path is `best_spreads.csv`. Existing output files are overwritten atomically, and the output parent directory must already exist. Call-side CSV columns are reserved for not-yet-implemented strategies and are blank for bull-put scans.

## Assumptions

- The tool uses live provider data and needs network access for real scans.
- Tradier option rows must include usable bid, ask, strike, and `greeks.mid_iv` values.
- Commodity scoring is optional and experimental. Enable it explicitly with `--use-commodity-score`; without it, the model uses a neutral commodity score.
- Commodity scoring uses FMP commodity history plus Yahoo Finance stock history through `yfinance`.
- Results are candidates for research, not trade recommendations.

## Troubleshooting

- `python spreadfinder.py --help` should work without API keys.
- Missing dependency: run `python -m pip install -r requirements.txt`.
- Missing optional dependency: install the optional package shown in the error message, or omit the feature that requires it.
- Bad or missing token: set `TRADIER_API_TOKEN` or pass `--api-token`. Authentication/provider failures exit nonzero instead of writing a misleading empty scan.
- No spreads found: widen DTE, lower `--min-ror`, lower `--min-prob-success`, or increase `--max-strike-dist`. Empty results can also come from illiquid chains or provider rows without usable bid/ask/strike/positive `greeks.mid_iv`; rerun with `--debug` for more detail.
- Rate limits or provider errors: retry later or reduce concurrent chain requests with `--batch-size`.

## Reserved Not Implemented

- Iron condor search.
- Plotting.
- Backtesting.
- Additional spread strategies.
