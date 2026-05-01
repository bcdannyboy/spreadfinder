#!/usr/bin/env python3
"""Find and rank bull put credit spreads from live option chains.

Quick start:
  python -m pip install -r requirements.txt
  python spreadfinder.py --self-test
  export TRADIER_API_TOKEN="..."
  python spreadfinder.py --symbols AAPL --min-dte 14 --max-dte 45

Run `python spreadfinder.py --help` for all options. The command remains a
single-file utility; optional market-data features are enabled only when used.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import logging
import math
import os
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any


VERSION = "0.2.1"

TRADIER_API_URL = "https://api.tradier.com/v1"
FMP_API_URL = "https://financialmodelingprep.com/api/v3"
DATE_FORMAT = "%Y-%m-%d"

RATE_LIMIT = 60
RATE_PERIOD_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_RETRIES = 2
TRANSIENT_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
TRADING_DAYS = 252
CALENDAR_DAYS = 365
CONTRACT_MULTIPLIER = 100
DEFAULT_VOLATILITY = 0.30
MAX_VOLATILITY = 2.0
NEUTRAL_COMMODITY_PROBABILITY = 0.50
DEFAULT_MAX_COMMODITIES = 25
MAX_WORKERS = min(8, os.cpu_count() or 1)
MAX_SIMULATION_CELLS = 5_000_000

CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

OUTPUT_COLUMNS = [
    "Symbol",
    "Spread Type",
    "Expiration",
    "DTE",
    "Underlying Price",
    "Volatility Used",
    "Short Put Strike",
    "Long Put Strike",
    "Short Put Distance From Spot",
    "Short Call Strike",
    "Long Call Strike",
    "Width",
    "Breakeven",
    "Credit",
    "Max Profit",
    "Max Loss",
    "Credit Per Contract",
    "Max Profit Per Contract",
    "Max Loss Per Contract",
    "Return on Risk",
    "Annualized ROR",
    "Probability of Success",
    "MC Profit Probability",
    "MC Max Profit Probability",
    "MC Heston Profit Probability",
    "Composite Probability",
    "Composite Score",
    "Expected Value",
    "Pricing State Put",
    "Pricing State Call",
    "Average Probability",
    "Short Put Bid Ask Spread",
    "Long Put Bid Ask Spread",
    "Min Volume",
    "Min Open Interest",
]

logger = logging.getLogger("spreadfinder")


class SpreadFinderHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


class RateLimiter:
    def __init__(self, calls: int, period_seconds: int) -> None:
        self.calls = calls
        self.period_seconds = period_seconds
        self.timestamps: deque[float] = deque()
        self.lock = Lock()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] >= self.period_seconds:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.calls:
                    self.timestamps.append(now)
                    return
                sleep_for = self.period_seconds - (now - self.timestamps[0])
            time.sleep(max(0.05, sleep_for))


TRADIER_LIMITER = RateLimiter(RATE_LIMIT, RATE_PERIOD_SECONDS)


@dataclass
class ScanConfig:
    symbols: list[str]
    mindte: int
    maxdte: int
    top_n: int
    min_ror: float
    max_strike_dist: float
    output: Path
    batch_size: int
    simulations: int
    risk_free_rate: float
    min_prob_success: float
    api_token: str
    commodities_api_key: str | None
    use_commodity_score: bool
    max_commodities: int


@dataclass
class SpreadResult:
    symbol: str
    spread_type: str
    expiration: str
    dte: int
    underlying_price: float
    volatility_used: float
    short_put_strike: float
    long_put_strike: float
    short_put_distance_from_spot: float
    short_call_strike: float | None
    long_call_strike: float | None
    width: float
    breakeven: float
    credit: float
    max_profit: float
    max_loss: float
    credit_per_contract: float
    max_profit_per_contract: float
    max_loss_per_contract: float
    return_on_risk: float
    annualized_ror: float
    probability_of_success: float
    mc_profit_probability: float
    mc_max_profit_probability: float
    mc_heston_profit_probability: float
    composite_probability: float
    composite_score: float
    expected_value: float
    pricing_state_put: str
    pricing_state_call: str | None
    average_probability: float
    short_put_bid_ask_spread: float
    long_put_bid_ask_spread: float
    min_volume: float
    min_open_interest: float


def require_module(module_name: str, package_name: str | None = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        package = package_name or module_name.split(".", 1)[0]
        raise RuntimeError(
            f"Missing dependency '{package}'. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_symbols(symbols: str | None) -> list[str]:
    if not symbols:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in symbols.split(","):
        symbol = item.strip().upper()
        if symbol and symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)
    return normalized


def tradier_headers(api_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_token}", "Accept": "application/json"}


def retry_delay_seconds(response: Any, attempt: int) -> float:
    retry_after = safe_float(getattr(response, "headers", {}).get("Retry-After") if response is not None else None)
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, 30.0)
    return min(0.5 * (2**attempt), 5.0)


def get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    context: str,
    expected_type: type | tuple[type, ...] = (dict, list),
) -> Any:
    requests = require_module("requests")
    response = None
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            if attempt < REQUEST_RETRIES:
                time.sleep(retry_delay_seconds(response, attempt))
                continue
            raise RuntimeError(f"{context} failed ({exc.__class__.__name__})") from None

        if response.status_code in TRANSIENT_HTTP_STATUSES and attempt < REQUEST_RETRIES:
            time.sleep(retry_delay_seconds(response, attempt))
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            reason = response.reason or "HTTP error"
            raise RuntimeError(f"{context} failed with HTTP {response.status_code}: {reason}") from None

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{context} returned invalid JSON") from exc
        break
    else:
        raise RuntimeError(f"{context} failed after retries")

    if not isinstance(payload, expected_type):
        raise RuntimeError(f"{context} returned an unexpected response shape")
    return payload


def get_stock_price(symbol: str, api_token: str) -> float:
    TRADIER_LIMITER.wait()
    data = get_json(
        f"{TRADIER_API_URL}/markets/quotes",
        headers=tradier_headers(api_token),
        params={"symbols": symbol},
        context=f"Tradier quote fetch for {symbol}",
        expected_type=dict,
    )
    quotes = data.get("quotes")
    if not isinstance(quotes, dict):
        raise RuntimeError(f"Tradier quote response for {symbol} did not include quotes")
    quote = quotes.get("quote")
    if isinstance(quote, list):
        quote = quote[0] if quote else None
    if not isinstance(quote, dict):
        raise RuntimeError(f"Tradier quote response for {symbol} did not include a quote")
    for field in ("last", "mark", "close"):
        price = safe_float(quote.get(field))
        if price is not None and price > 0:
            return price
    raise RuntimeError(f"Tradier quote response for {symbol} did not include a usable price")


def get_option_expirations(symbol: str, api_token: str) -> list[str]:
    TRADIER_LIMITER.wait()
    data = get_json(
        f"{TRADIER_API_URL}/markets/options/expirations",
        headers=tradier_headers(api_token),
        params={"symbol": symbol},
        context=f"Tradier expiration fetch for {symbol}",
        expected_type=dict,
    )
    expirations_data = data.get("expirations")
    if not isinstance(expirations_data, dict):
        raise RuntimeError(f"Tradier expiration response for {symbol} did not include expirations")
    expirations = expirations_data.get("date")
    return [str(exp) for exp in as_list(expirations)]


def get_option_chain(symbol: str, expiration: str, api_token: str) -> list[dict[str, Any]]:
    TRADIER_LIMITER.wait()
    data = get_json(
        f"{TRADIER_API_URL}/markets/options/chains",
        headers=tradier_headers(api_token),
        params={"symbol": symbol, "expiration": expiration, "greeks": "true"},
        context=f"Tradier option-chain fetch for {symbol} {expiration}",
        expected_type=dict,
    )
    options = data.get("options")
    if not isinstance(options, dict):
        raise RuntimeError(f"Tradier option-chain response for {symbol} {expiration} did not include options")
    option_rows = as_list(options.get("option"))
    puts = [row for row in option_rows if isinstance(row, dict) and row.get("option_type") == "put"]
    if not puts:
        logger.warning("No put option data found for %s %s", symbol, expiration)
    return puts


def get_historical_volatility(symbol: str, api_token: str, cache: dict[str, float], days: int = TRADING_DAYS) -> float:
    cache_key = f"{symbol}_{days}"
    if cache_key in cache:
        return cache[cache_key]

    end = dt.date.today()
    start = end - dt.timedelta(days=CALENDAR_DAYS)
    try:
        TRADIER_LIMITER.wait()
        data = get_json(
            f"{TRADIER_API_URL}/markets/history",
            headers=tradier_headers(api_token),
            params={
                "symbol": symbol,
                "interval": "daily",
                "start": start.strftime(DATE_FORMAT),
                "end": end.strftime(DATE_FORMAT),
            },
            context=f"Tradier history fetch for {symbol}",
            expected_type=dict,
        )
        history = as_list(data.get("history", {}).get("day"))
        closes = [safe_float(day.get("close")) for day in history if isinstance(day, dict)]
        closes = [close for close in closes if close is not None and close > 0]
        volatility = estimate_annualized_volatility(closes, days)
    except RuntimeError as exc:
        logger.warning("%s; using default volatility %.0f%%", exc, DEFAULT_VOLATILITY * 100)
        volatility = DEFAULT_VOLATILITY

    cache[cache_key] = volatility
    return volatility


def estimate_annualized_volatility(closes: list[float], days: int = TRADING_DAYS) -> float:
    np = require_module("numpy")
    if len(closes) < 30:
        logger.warning("Insufficient price history for GARCH; using simple volatility fallback")
        return simple_annualized_volatility(closes, days)

    prices = np.array(closes, dtype=float)
    log_returns = np.diff(np.log(prices))
    if len(log_returns) < 20:
        return simple_annualized_volatility(closes, days)

    try:
        arch = require_module("arch", "arch")
        scaled_returns = log_returns * 100.0
        model = arch.arch_model(scaled_returns, vol="Garch", p=1, q=1, rescale=False)
        result = model.fit(disp="off")
        forecast = result.forecast(horizon=1)
        daily_volatility = math.sqrt(float(forecast.variance.values[-1, 0])) / 100.0
    except RuntimeError:
        logger.warning("arch is not installed; using simple volatility fallback")
        daily_volatility = float(np.std(log_returns, ddof=1))
    except Exception as exc:
        logger.warning("GARCH volatility failed (%s); using simple volatility fallback", exc)
        daily_volatility = float(np.std(log_returns, ddof=1))

    annualized = daily_volatility * math.sqrt(days)
    return clamp(annualized, 0.01, MAX_VOLATILITY)


def simple_annualized_volatility(closes: list[float], days: int = TRADING_DAYS) -> float:
    np = require_module("numpy")
    if len(closes) < 3:
        return DEFAULT_VOLATILITY
    prices = np.array(closes, dtype=float)
    log_returns = np.diff(np.log(prices))
    if len(log_returns) < 2:
        return DEFAULT_VOLATILITY
    annualized = float(np.std(log_returns, ddof=1)) * math.sqrt(days)
    return clamp(annualized, 0.01, MAX_VOLATILITY)


def fetch_commodity_data(api_key: str, symbol: str) -> dict[str, Any] | None:
    try:
        return get_json(
            f"{FMP_API_URL}/historical-price-full/{symbol}",
            params={"apikey": api_key},
            context=f"FMP commodity fetch for {symbol}",
            expected_type=dict,
        )
    except RuntimeError as exc:
        logger.warning("%s", exc)
        return None


def fetch_stock_data(ticker: str, period: str = "5y") -> Any:
    pd = require_module("pandas")
    try:
        yf = importlib.import_module("yfinance")
    except ImportError:
        logger.warning("Optional dependency 'yfinance' is not installed; commodity score will be neutral.")
        return pd.DataFrame()
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            logger.warning("No Yahoo Finance data found for %s", ticker)
            return pd.DataFrame()
        hist.reset_index(inplace=True)
        hist["Date"] = hist["Date"].dt.tz_localize(None)
        hist = hist[["Date", "Close"]]
        hist["Close"] = pd.to_numeric(hist["Close"], errors="coerce")
        hist.dropna(inplace=True)
        return hist
    except Exception as exc:
        logger.warning("Error fetching Yahoo Finance data for %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_commodities_list(api_key: str) -> dict[str, str]:
    data = get_json(
        f"{FMP_API_URL}/symbol/available-commodities",
        params={"apikey": api_key},
        context="FMP commodities list fetch",
        expected_type=list,
    )
    commodities = as_list(data)
    return {
        str(item.get("symbol")): str(item.get("name", item.get("symbol")))
        for item in commodities
        if isinstance(item, dict) and item.get("symbol")
    }


def calculate_correlations(stock_data: Any, commodities_data: dict[str, dict[str, Any]]) -> dict[str, float]:
    pd = require_module("pandas")
    np = require_module("numpy")
    if stock_data.empty or "Date" not in stock_data.columns or "Close" not in stock_data.columns:
        logger.warning("Stock data is empty. Commodity score will be neutral.")
        return {}

    stock_close = stock_data[["Date", "Close"]].copy()
    stock_close["Date"] = pd.to_datetime(stock_close["Date"], errors="coerce")
    stock_close.dropna(subset=["Date", "Close"], inplace=True)
    stock_close.set_index("Date", inplace=True)
    stock_close["Close"] = pd.to_numeric(stock_close["Close"], errors="coerce")

    correlations: dict[str, float] = {}
    for symbol, data in commodities_data.items():
        historical = as_list(data.get("historical")) if isinstance(data, dict) else []
        if not historical:
            continue
        df = pd.DataFrame(historical)
        if "date" not in df.columns or "close" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date", "close"], inplace=True)
        df.set_index("date", inplace=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        combined = stock_close.join(df[["close"]], how="inner")
        combined.dropna(inplace=True)
        combined = combined[(combined["Close"] > 0) & (combined["close"] > 0)]
        if len(combined) >= 3:
            returns = np.log(combined[["Close", "close"]]).diff().dropna()
            corr = safe_float(returns["Close"].corr(returns["close"])) if len(returns) >= 2 else None
            if corr is not None:
                correlations[symbol] = corr
    return correlations


def commodity_probability(correlations: dict[str, float] | float | None) -> float:
    if correlations is None:
        return NEUTRAL_COMMODITY_PROBABILITY
    if isinstance(correlations, (int, float)):
        probability = safe_float(correlations)
        return NEUTRAL_COMMODITY_PROBABILITY if probability is None else clamp(probability)

    values = [safe_float(value) for value in correlations.values()]
    normalized = [clamp((value + 1.0) / 2.0) for value in values if value is not None]
    if not normalized:
        return NEUTRAL_COMMODITY_PROBABILITY
    return sum(normalized) / len(normalized)


def calculate_commodity_probability(symbol: str, api_key: str | None, max_commodities: int) -> float:
    if not api_key:
        return NEUTRAL_COMMODITY_PROBABILITY
    try:
        commodities_list = fetch_commodities_list(api_key)
    except RuntimeError as exc:
        logger.warning("%s; commodity score will be neutral", exc)
        return NEUTRAL_COMMODITY_PROBABILITY

    symbols = list(commodities_list.keys())[:max_commodities]
    if not symbols:
        return NEUTRAL_COMMODITY_PROBABILITY

    commodities_data: dict[str, dict[str, Any]] = {}
    workers = min(MAX_WORKERS, len(symbols))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_commodity_data, api_key, commodity): commodity for commodity in symbols}
        for future in as_completed(futures):
            commodity = futures[future]
            try:
                data = future.result()
            except Exception as exc:
                logger.warning("Commodity fetch failed for %s: %s", commodity, exc)
                continue
            if data:
                commodities_data[commodity] = data

    if not commodities_data:
        return NEUTRAL_COMMODITY_PROBABILITY
    try:
        stock_data = fetch_stock_data(symbol)
        return commodity_probability(calculate_correlations(stock_data, commodities_data))
    except Exception as exc:
        logger.warning("Commodity correlation failed for %s (%s); commodity score will be neutral", symbol, exc)
        return NEUTRAL_COMMODITY_PROBABILITY


def black_scholes_price(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
    if option_type == "put":
        return K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)
    raise ValueError(f"Unsupported option_type: {option_type}")


def american_option_binomial(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", steps: int = 100) -> float:
    np = require_module("numpy")
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)

    steps = max(1, int(steps))
    dt_step = T / steps
    u = math.exp(sigma * math.sqrt(dt_step))
    d = 1.0 / u
    denom = u - d
    if abs(denom) < 1e-12:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    p = clamp((math.exp(r * dt_step) - d) / denom)

    terminal_nodes = np.arange(steps + 1)
    asset_prices = S * (u ** (steps - terminal_nodes)) * (d**terminal_nodes)
    if option_type == "call":
        option_values = np.maximum(0, asset_prices - K)
    elif option_type == "put":
        option_values = np.maximum(0, K - asset_prices)
    else:
        raise ValueError(f"Unsupported option_type: {option_type}")

    discount = math.exp(-r * dt_step)
    for i in range(steps - 1, -1, -1):
        option_values = discount * (p * option_values[:-1] + (1.0 - p) * option_values[1:])
        nodes = np.arange(i + 1)
        asset_prices = S * (u ** (i - nodes)) * (d**nodes)
        exercise = np.maximum(0, asset_prices - K) if option_type == "call" else np.maximum(0, K - asset_prices)
        option_values = np.maximum(option_values, exercise)
    return float(option_values[0])


def probability_above_threshold(S: float, threshold: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or threshold <= 0:
        return 1.0 if S > threshold else 0.0
    d2 = (math.log(S / threshold) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return clamp(normal_cdf(d2))


def monte_carlo_terminal_prices(
    current_price: float,
    days: int,
    simulations: int,
    volatility: float,
    *,
    use_t_dist: bool = False,
    df: int = 3,
    use_heston: bool = False,
    risk_free_rate: float = 0.0,
    seed: int | None = None,
    kappa: float = 2.0,
    theta: float = 0.02,
    xi: float = 0.1,
    rho: float = -0.7,
) -> Any:
    np = require_module("numpy")
    steps = max(1, int(days))
    simulations = max(1, int(simulations))
    if steps * simulations > MAX_SIMULATION_CELLS:
        raise RuntimeError(
            f"Monte Carlo request is too large ({steps * simulations:,} cells); "
            f"reduce --simulations or --max-dte"
        )
    dt_step = (steps / CALENDAR_DAYS) / steps

    try:
        stats = require_module("scipy.stats", "scipy")
        qmc = require_module("scipy.stats.qmc", "scipy")
        sampler = qmc.LatinHypercube(d=steps, seed=seed)
        samples = sampler.random(n=simulations)
        shocks = stats.t.ppf(samples, df) if use_t_dist else stats.norm.ppf(samples)
    except RuntimeError:
        rng = np.random.default_rng(seed)
        shocks = rng.standard_t(df, size=(simulations, steps)) if use_t_dist else rng.standard_normal((simulations, steps))

    shocks = np.nan_to_num(shocks, nan=0.0, posinf=5.0, neginf=-5.0)
    if use_t_dist and df > 2:
        shocks = shocks / math.sqrt(df / (df - 2))
    volatility = max(0.0001, min(float(volatility), MAX_VOLATILITY))

    if use_heston:
        rng = np.random.default_rng(None if seed is None else seed + 1)
        variance = np.full(simulations, volatility**2)
        prices = np.full(simulations, current_price, dtype=float)
        z1 = shocks.T
        z2 = rng.standard_normal(size=(steps, simulations))
        z2 = rho * z1 + math.sqrt(max(0.0, 1.0 - rho**2)) * z2
        for day in range(steps):
            variance = np.maximum(
                0.0,
                variance + kappa * (theta - variance) * dt_step + xi * np.sqrt(variance) * math.sqrt(dt_step) * z1[day],
            )
            variance = np.minimum(variance, 4.0 * volatility**2)
            prices *= np.exp((risk_free_rate - 0.5 * variance) * dt_step + np.sqrt(variance * dt_step) * z2[day])
        return prices

    log_returns = (risk_free_rate - 0.5 * volatility**2) * dt_step + volatility * math.sqrt(dt_step) * shocks
    return current_price * np.exp(np.sum(log_returns, axis=1))


def fetch_option_chains(
    symbols: list[str],
    mindte: int,
    maxdte: int,
    api_token: str,
    batch_size: int = 10,
) -> tuple[list[tuple[str, str, Any]], list[str]]:
    pd = require_module("pandas")
    all_puts: list[tuple[str, str, Any]] = []
    errors: list[str] = []
    today = dt.date.today()

    for symbol in symbols:
        logger.info("Fetching expirations for %s", symbol)
        try:
            expirations = get_option_expirations(symbol, api_token)
        except RuntimeError as exc:
            logger.error("%s", exc)
            errors.append(str(exc))
            continue

        valid_expirations: list[str] = []
        for expiration in expirations:
            try:
                expiration_date = dt.datetime.strptime(expiration, DATE_FORMAT).date()
            except ValueError:
                logger.warning("Skipping invalid expiration date from provider: %s", expiration)
                continue
            dte = (expiration_date - today).days
            if mindte <= dte <= maxdte:
                valid_expirations.append(expiration)

        if not valid_expirations:
            logger.warning("No expirations found for %s in DTE range %s-%s", symbol, mindte, maxdte)
            continue

        workers = min(max(1, batch_size), MAX_WORKERS, len(valid_expirations))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(get_option_chain, symbol, expiration, api_token): expiration
                for expiration in valid_expirations
            }
            for future in as_completed(futures):
                expiration = futures[future]
                try:
                    puts = future.result()
                except RuntimeError as exc:
                    logger.warning("%s", exc)
                    errors.append(str(exc))
                    continue
                if puts:
                    all_puts.append((symbol, expiration, pd.DataFrame(puts)))

        symbol_put_count = sum(1 for item in all_puts if item[0] == symbol)
        logger.info("Found %s put chains for %s", symbol_put_count, symbol)
    return all_puts, errors


def extract_mid_iv(greeks: Any) -> float | None:
    return safe_float(greeks.get("mid_iv")) if isinstance(greeks, dict) else None


def prepare_put_chain(put_chain: Any) -> Any:
    pd = require_module("pandas")
    if put_chain is None or put_chain.empty:
        return pd.DataFrame()

    df = put_chain.copy()
    if "greeks" not in df.columns:
        df["mid_iv"] = None
    else:
        df["mid_iv"] = df["greeks"].apply(extract_mid_iv)

    for column in ("strike", "bid", "ask", "volume", "open_interest", "mid_iv"):
        if column not in df.columns:
            df[column] = 0 if column in ("volume", "open_interest") else None
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["strike", "bid", "ask", "mid_iv"])
    df = df[(df["strike"] > 0) & (df["bid"] >= 0) & (df["ask"] >= 0) & (df["ask"] >= df["bid"]) & (df["mid_iv"] > 0)]
    if df.empty:
        return df

    df["bid_ask_spread"] = df["ask"] - df["bid"]
    df["volume"] = df["volume"].fillna(0)
    df["open_interest"] = df["open_interest"].fillna(0)
    df = df.sort_values("strike").drop_duplicates(subset=["strike"], keep="last").reset_index(drop=True)
    return df


def classify_pricing_state(market_credit: float, theoretical_credit: float) -> str:
    tolerance = max(0.01, abs(theoretical_credit) * 0.05)
    if market_credit > theoretical_credit + tolerance:
        return "overpriced"
    if market_credit < theoretical_credit - tolerance:
        return "underpriced"
    return "fairly priced"


def expected_bull_put_value(final_prices: Any, short_strike: float, long_strike: float, credit: float) -> float:
    np = require_module("numpy")
    payoff = credit - np.maximum(short_strike - final_prices, 0.0) + np.maximum(long_strike - final_prices, 0.0)
    return float(np.mean(payoff))


def composite_probability(
    probability_success: float,
    mc_profit_probability: float,
    mc_max_profit_probability: float,
    mc_heston_profit_probability: float,
    commodity_prob: float,
) -> float:
    return clamp(
        0.30 * probability_success
        + 0.30 * mc_profit_probability
        + 0.20 * mc_max_profit_probability
        + 0.10 * mc_heston_profit_probability
        + 0.10 * commodity_prob
    )


def score_spread(
    composite_prob: float,
    return_on_risk: float,
    expected_value: float,
    width: float,
    credit: float,
    liquidity_penalty: float,
    pricing_state: str,
) -> float:
    ror_component = clamp(return_on_risk / 0.50)
    ev_component = clamp((expected_value / max(width, 0.01)) + 0.5)
    credit_component = clamp(credit / max(width, 0.01))
    liquidity_component = clamp(1.0 - liquidity_penalty)
    pricing_bonus = 0.04 if pricing_state == "overpriced" else (-0.04 if pricing_state == "underpriced" else 0.0)
    return clamp(
        0.38 * composite_prob
        + 0.24 * ror_component
        + 0.18 * ev_component
        + 0.10 * credit_component
        + 0.10 * liquidity_component
        + pricing_bonus
    )


def process_bull_put_spread(
    symbol: str,
    expiration: str,
    put_chain: Any,
    underlying_price: float,
    min_ror: float,
    max_strike_dist: float,
    simulations: int,
    volatility: float,
    risk_free_rate: float,
    min_prob_success: float,
    commodity_prob: float,
    seed: int | None = None,
) -> list[SpreadResult]:
    pd = require_module("pandas")
    np = require_module("numpy")

    try:
        expiration_date = dt.datetime.strptime(expiration, DATE_FORMAT).date()
    except ValueError:
        logger.warning("Skipping invalid expiration date: %s", expiration)
        return []

    dte = (expiration_date - dt.date.today()).days
    if dte <= 0:
        return []

    df = prepare_put_chain(put_chain)
    if df.empty or len(df) < 2:
        return []

    T = dte / CALENDAR_DAYS
    df = df.copy()
    df["bs_price"] = [
        black_scholes_price("put", underlying_price, row.strike, T, risk_free_rate, row.mid_iv)
        for row in df.itertuples()
    ]
    df["binomial_price"] = [
        american_option_binomial(underlying_price, row.strike, T, risk_free_rate, row.mid_iv, option_type="put")
        for row in df.itertuples()
    ]
    df["theoretical_price"] = (df["bs_price"] + df["binomial_price"]) / 2.0

    final_prices = monte_carlo_terminal_prices(
        underlying_price,
        dte,
        simulations,
        volatility,
        use_t_dist=True,
        risk_free_rate=0.0,
        seed=seed,
    )
    heston_final_prices = monte_carlo_terminal_prices(
        underlying_price,
        dte,
        simulations,
        volatility,
        use_heston=True,
        risk_free_rate=risk_free_rate,
        seed=seed,
    )

    results: list[SpreadResult] = []
    for short_index, short_put in df.iterrows():
        short_strike = float(short_put["strike"])
        if short_strike > underlying_price * (1 + max_strike_dist) or short_strike < underlying_price * (1 - max_strike_dist):
            continue

        long_candidates = df.loc[: short_index - 1] if short_index > 0 else pd.DataFrame()
        for _, long_put in long_candidates.iterrows():
            long_strike = float(long_put["strike"])
            if short_strike <= long_strike:
                continue

            credit = float(short_put["bid"]) - float(long_put["ask"])
            if credit <= 0:
                continue
            width = short_strike - long_strike
            max_loss = width - credit
            if max_loss <= 0:
                continue

            return_on_risk = credit / max_loss
            if return_on_risk < min_ror:
                continue

            breakeven = short_strike - credit
            probability_success = probability_above_threshold(
                underlying_price,
                breakeven,
                T,
                risk_free_rate,
                float(short_put["mid_iv"]),
            )
            mc_profit_probability = float(np.mean(final_prices > breakeven))
            mc_max_profit_probability = float(np.mean(final_prices > short_strike))
            mc_heston_profit_probability = float(np.mean(heston_final_prices > breakeven))
            average_probability = (probability_success + mc_profit_probability + mc_heston_profit_probability) / 3.0

            comp_probability = composite_probability(
                probability_success,
                mc_profit_probability,
                mc_max_profit_probability,
                mc_heston_profit_probability,
                commodity_prob,
            )
            if comp_probability < min_prob_success:
                continue

            theoretical_credit = float(short_put["theoretical_price"]) - float(long_put["theoretical_price"])
            pricing_state = classify_pricing_state(credit, theoretical_credit)
            expected_value = expected_bull_put_value(final_prices, short_strike, long_strike, credit)

            short_spread = float(short_put["bid_ask_spread"])
            long_spread = float(long_put["bid_ask_spread"])
            liquidity_penalty = clamp((short_spread + long_spread) / max(width, 0.01))
            composite = score_spread(
                comp_probability,
                return_on_risk,
                expected_value,
                width,
                credit,
                liquidity_penalty,
                pricing_state,
            )
            annualized_ror = (1.0 + return_on_risk) ** (CALENDAR_DAYS / dte) - 1.0

            min_volume = min(float(short_put["volume"]), float(long_put["volume"]))
            min_open_interest = min(float(short_put["open_interest"]), float(long_put["open_interest"]))

            results.append(
                SpreadResult(
                    symbol=symbol,
                    spread_type="bull_put",
                    expiration=expiration,
                    dte=dte,
                    underlying_price=underlying_price,
                    volatility_used=volatility,
                    short_put_strike=short_strike,
                    long_put_strike=long_strike,
                    short_put_distance_from_spot=(short_strike / underlying_price) - 1.0,
                    short_call_strike=None,
                    long_call_strike=None,
                    width=width,
                    breakeven=breakeven,
                    credit=credit,
                    max_profit=credit,
                    max_loss=max_loss,
                    credit_per_contract=credit * CONTRACT_MULTIPLIER,
                    max_profit_per_contract=credit * CONTRACT_MULTIPLIER,
                    max_loss_per_contract=max_loss * CONTRACT_MULTIPLIER,
                    return_on_risk=return_on_risk,
                    annualized_ror=annualized_ror,
                    probability_of_success=probability_success,
                    mc_profit_probability=mc_profit_probability,
                    mc_max_profit_probability=mc_max_profit_probability,
                    mc_heston_profit_probability=mc_heston_profit_probability,
                    composite_probability=comp_probability,
                    composite_score=composite,
                    expected_value=expected_value,
                    pricing_state_put=pricing_state,
                    pricing_state_call=None,
                    average_probability=average_probability,
                    short_put_bid_ask_spread=short_spread,
                    long_put_bid_ask_spread=long_spread,
                    min_volume=min_volume,
                    min_open_interest=min_open_interest,
                )
            )

    logger.info("Scored %s bull put spreads for %s %s", len(results), symbol, expiration)
    return results


def find_bull_put_spreads(
    puts: list[tuple[str, str, Any]],
    underlying_price: float,
    min_ror: float,
    max_strike_dist: float,
    simulations: int,
    volatility: float,
    risk_free_rate: float,
    min_prob_success: float,
    commodity_prob: float,
) -> list[SpreadResult]:
    spreads: list[SpreadResult] = []
    for symbol, expiration, put_chain in puts:
        spreads.extend(
            process_bull_put_spread(
                symbol,
                expiration,
                put_chain,
                underlying_price,
                min_ror,
                max_strike_dist,
                simulations,
                volatility,
                risk_free_rate,
                min_prob_success,
                commodity_prob,
            )
        )
    return spreads


def spread_to_row(spread: SpreadResult) -> dict[str, Any]:
    return {
        "Symbol": spread.symbol,
        "Spread Type": spread.spread_type,
        "Expiration": spread.expiration,
        "DTE": spread.dte,
        "Underlying Price": spread.underlying_price,
        "Volatility Used": spread.volatility_used,
        "Short Put Strike": spread.short_put_strike,
        "Long Put Strike": spread.long_put_strike,
        "Short Put Distance From Spot": spread.short_put_distance_from_spot,
        "Short Call Strike": spread.short_call_strike,
        "Long Call Strike": spread.long_call_strike,
        "Width": spread.width,
        "Breakeven": spread.breakeven,
        "Credit": spread.credit,
        "Max Profit": spread.max_profit,
        "Max Loss": spread.max_loss,
        "Credit Per Contract": spread.credit_per_contract,
        "Max Profit Per Contract": spread.max_profit_per_contract,
        "Max Loss Per Contract": spread.max_loss_per_contract,
        "Return on Risk": spread.return_on_risk,
        "Annualized ROR": spread.annualized_ror,
        "Probability of Success": spread.probability_of_success,
        "MC Profit Probability": spread.mc_profit_probability,
        "MC Max Profit Probability": spread.mc_max_profit_probability,
        "MC Heston Profit Probability": spread.mc_heston_profit_probability,
        "Composite Probability": spread.composite_probability,
        "Composite Score": spread.composite_score,
        "Expected Value": spread.expected_value,
        "Pricing State Put": spread.pricing_state_put,
        "Pricing State Call": spread.pricing_state_call,
        "Average Probability": spread.average_probability,
        "Short Put Bid Ask Spread": spread.short_put_bid_ask_spread,
        "Long Put Bid Ask Spread": spread.long_put_bid_ask_spread,
        "Min Volume": spread.min_volume,
        "Min Open Interest": spread.min_open_interest,
    }


def sanitize_csv_cell(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(CSV_FORMULA_PREFIXES):
        return "'" + value
    return value


def write_results_csv(spreads: list[SpreadResult], output: Path) -> None:
    import csv

    rows = [{key: sanitize_csv_cell(value) for key, value in spread_to_row(spread).items()} for spread in spreads]
    output = output.expanduser()
    tmp_name = f".{output.name}.tmp"
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=output.parent, prefix=tmp_name, delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(tmp.name)
    try:
        os.replace(temp_path, output)
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise


def render_results(spreads: list[SpreadResult], output: Path, top_n: int) -> None:
    if not spreads:
        print(f"No spreads matched the filters. Wrote header-only CSV to {output}.")
        print("Try widening DTE, lowering --min-ror, lowering --min-prob-success, or increasing --max-strike-dist.")
        return

    print(f"Top {min(top_n, len(spreads))} spreads:")
    for spread in spreads:
        print(
            f"{spread.symbol} {spread.expiration} {spread.spread_type}: "
            f"DTE {spread.dte}, sell {spread.short_put_strike:g}P / buy {spread.long_put_strike:g}P, "
            f"credit {spread.credit:.2f} per share (${spread.credit_per_contract:.0f}/contract), "
            f"breakeven {spread.breakeven:.2f}, "
            f"max loss ${spread.max_loss_per_contract:.0f}/contract, "
            f"ROR {spread.return_on_risk * 100:.1f}%, "
            f"profit probability {spread.mc_profit_probability * 100:.1f}%, "
            f"composite probability {spread.composite_probability * 100:.1f}%, "
            f"EV {spread.expected_value:.2f}, "
            f"liq min vol/OI {spread.min_volume:.0f}/{spread.min_open_interest:.0f}, "
            f"score {spread.composite_score:.3f}"
        )
    print(f"Wrote CSV to {output}.")


def run_scan(config: ScanConfig) -> list[SpreadResult]:
    start_time = dt.datetime.now()
    logger.info("Fetching option chains for %s", ", ".join(config.symbols))
    puts, scan_errors = fetch_option_chains(config.symbols, config.mindte, config.maxdte, config.api_token, config.batch_size)
    if scan_errors and not puts:
        raise RuntimeError("No usable option chains fetched; last provider error: " + scan_errors[-1])

    commodity_scores: dict[str, float] = {}
    if config.use_commodity_score:
        logger.info("Fetching optional commodity correlation score")
        for symbol in config.symbols:
            commodity_scores[symbol] = calculate_commodity_probability(symbol, config.commodities_api_key, config.max_commodities)
    else:
        commodity_scores = {symbol: NEUTRAL_COMMODITY_PROBABILITY for symbol in config.symbols}

    all_spreads: list[SpreadResult] = []
    processed_symbols = 0
    cache: dict[str, float] = {}
    for symbol in config.symbols:
        symbol_puts = [put for put in puts if put[0] == symbol]
        if not symbol_puts:
            logger.warning("No put chains available for %s after expiration filtering", symbol)
            continue
        try:
            logger.info("Fetching quote and volatility for %s", symbol)
            underlying_price = get_stock_price(symbol, config.api_token)
            volatility = get_historical_volatility(symbol, config.api_token, cache)
            processed_symbols += 1
            spreads = find_bull_put_spreads(
                symbol_puts,
                underlying_price,
                config.min_ror,
                config.max_strike_dist,
                config.simulations,
                volatility,
                config.risk_free_rate,
                config.min_prob_success,
                commodity_scores.get(symbol, NEUTRAL_COMMODITY_PROBABILITY),
            )
        except RuntimeError as exc:
            message = f"{symbol}: {exc}"
            logger.warning("Skipping %s", message)
            scan_errors.append(message)
            continue
        all_spreads.extend(spreads)

    if scan_errors and not all_spreads and processed_symbols == 0:
        raise RuntimeError("No usable symbols scored; last error: " + scan_errors[-1])

    all_spreads.sort(key=lambda spread: spread.composite_score, reverse=True)
    best_spreads = all_spreads[: config.top_n]
    write_results_csv(best_spreads, config.output)
    render_results(best_spreads, config.output, config.top_n)
    logger.info("Time taken: %s", dt.datetime.now() - start_time)
    return best_spreads


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find and rank bull put credit spreads from Tradier option chains.",
        formatter_class=SpreadFinderHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python spreadfinder.py --self-test\n"
            "  python spreadfinder.py --symbols AAPL --min-dte 14 --max-dte 45\n"
            "  python spreadfinder.py --symbols AAPL,MSFT --min-dte 21 --max-dte 60 --min-ror 0.20 --output spreads.csv\n"
            "Set TRADIER_API_TOKEN in your environment, or pass --api-token for ad hoc scans."
        ),
    )
    parser.add_argument("--symbols", "-s", "-symbols", dest="symbols", help="Comma-separated symbols, for example AAPL,MSFT")
    parser.add_argument("--min-dte", "-m", "-mindte", dest="mindte", type=int, help="Minimum days to expiration")
    parser.add_argument("--max-dte", "-l", "-maxdte", dest="maxdte", type=int, help="Maximum days to expiration")
    parser.add_argument("--top", "--top-n", "-n", "-top_n", dest="top_n", type=int, default=10, help="Number of top spreads to display")
    parser.add_argument("--min-ror", "-r", "-min_ror", dest="min_ror", type=float, default=0.15, help="Minimum return on risk as a decimal; 0.15 means 15%%")
    parser.add_argument("--max-strike-dist", "-d", "-max_strike_dist", dest="max_strike_dist", type=float, default=0.2, help="Max short-strike distance from spot as a decimal; 0.2 means 20%%")
    parser.add_argument("--output", "-o", "-output", dest="output", default="best_spreads.csv", help="Output CSV file path")
    parser.add_argument("--batch-size", "-b", "-batch_size", dest="batch_size", type=int, default=10, help="Max concurrent option-chain requests")
    parser.add_argument("--include-iron-condors", "--include_iron_condors", action="store_true", help="Reserved; currently fails fast because iron condors are not implemented")
    parser.add_argument("--api-token", "-api_token", dest="api_token", help="Tradier API token. Prefer TRADIER_API_TOKEN.")
    parser.add_argument("--simulations", "-sim", "-simulations", dest="simulations", type=int, default=1000, help="Monte Carlo simulations per expiration")
    parser.add_argument("--risk-free-rate", "-rf", "-risk_free_rate", dest="risk_free_rate", type=float, default=0.01, help="Risk-free rate as a decimal; 0.01 means 1%%")
    parser.add_argument("--plot", action="store_true", help="Reserved; currently fails fast because plotting is not implemented")
    parser.add_argument("--backtesting", action="store_true", help="Reserved; currently fails fast because backtesting is not implemented")
    parser.add_argument("--min-prob-success", "-ps", "-min_prob_success", dest="min_prob_success", type=float, default=0.5, help="Minimum composite probability as a decimal")
    parser.add_argument("--commodities-api-key", "-c", "-commodities_api_key", dest="commodities_api_key", help="FinancialModelingPrep API key. Prefer FMP_API_KEY.")
    parser.add_argument("--use-commodity-score", action="store_true", help="Fetch FMP commodity data and include it in the composite score")
    parser.add_argument("--max-commodities", type=int, default=DEFAULT_MAX_COMMODITIES, help="Max FMP commodities sampled when commodity scoring is enabled")
    parser.add_argument("--self-test", action="store_true", help="Run a no-network smoke test and exit")
    parser.add_argument("--quiet", action="store_true", help="Only print final results and errors")
    parser.add_argument("--debug", action="store_true", help="Show debug logs and tracebacks")
    parser.add_argument("--version", action="version", version=f"SpreadFinder {VERSION}")
    return parser


def configure_logging(args: argparse.Namespace) -> None:
    level = logging.DEBUG if args.debug else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    logging.getLogger("numexpr").setLevel(logging.WARNING)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.include_iron_condors:
        parser.error("--include-iron-condors is not implemented yet; omit it for the supported bull-put scan")
    if args.plot:
        parser.error("--plot is not implemented yet; CSV output remains available")
    if args.backtesting:
        parser.error("--backtesting is not implemented yet")

    if args.symbols is not None:
        args.symbols = normalize_symbols(args.symbols)
        if not args.symbols:
            parser.error("--symbols must include at least one non-empty symbol")

    if args.mindte is not None and args.mindte < 1:
        parser.error("--min-dte must be at least 1")
    if args.maxdte is not None and args.maxdte < 1:
        parser.error("--max-dte must be at least 1")
    if args.mindte is not None and args.maxdte is not None and args.mindte > args.maxdte:
        parser.error("--min-dte cannot be greater than --max-dte")

    if args.top_n <= 0:
        parser.error("--top must be greater than 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    if args.simulations <= 0:
        parser.error("--simulations must be greater than 0")
    for attr, flag in (
        ("min_ror", "--min-ror"),
        ("max_strike_dist", "--max-strike-dist"),
        ("risk_free_rate", "--risk-free-rate"),
        ("min_prob_success", "--min-prob-success"),
    ):
        value = getattr(args, attr)
        if not math.isfinite(value):
            parser.error(f"{flag} must be finite")
    if args.min_ror < 0:
        parser.error("--min-ror cannot be negative")
    if not 0 <= args.max_strike_dist <= 1:
        parser.error("--max-strike-dist must be between 0 and 1")
    if not 0 <= args.min_prob_success <= 1:
        parser.error("--min-prob-success must be between 0 and 1")
    if args.max_commodities <= 0:
        parser.error("--max-commodities must be greater than 0")

    if args.self_test:
        return

    if not args.symbols:
        parser.error("--symbols is required unless --self-test is used")
    if args.mindte is None:
        parser.error("--min-dte is required unless --self-test is used")
    if args.maxdte is None:
        parser.error("--max-dte is required unless --self-test is used")
    if args.simulations * args.maxdte > MAX_SIMULATION_CELLS:
        parser.error(
            f"--simulations times --max-dte must be at most {MAX_SIMULATION_CELLS:,} "
            "to avoid excessive local memory use"
        )

    output = Path(args.output).expanduser()
    if output.exists() and output.is_dir():
        parser.error("--output must be a file path, not a directory")
    if not output.parent.exists():
        parser.error(f"--output parent directory does not exist: {output.parent}")
    args.output = output

    args.api_token = args.api_token or os.getenv("TRADIER_API_TOKEN")
    if not args.api_token:
        parser.error("Tradier API token required; pass --api-token or set TRADIER_API_TOKEN")

    args.commodities_api_key = args.commodities_api_key or os.getenv("FMP_API_KEY")
    if args.use_commodity_score and not args.commodities_api_key:
        parser.error("Commodity scoring requires --commodities-api-key or FMP_API_KEY")


def config_from_args(args: argparse.Namespace) -> ScanConfig:
    return ScanConfig(
        symbols=args.symbols,
        mindte=args.mindte,
        maxdte=args.maxdte,
        top_n=args.top_n,
        min_ror=args.min_ror,
        max_strike_dist=args.max_strike_dist,
        output=args.output,
        batch_size=args.batch_size,
        simulations=args.simulations,
        risk_free_rate=args.risk_free_rate,
        min_prob_success=args.min_prob_success,
        api_token=args.api_token,
        commodities_api_key=args.commodities_api_key,
        use_commodity_score=args.use_commodity_score,
        max_commodities=args.max_commodities,
    )


def run_self_test() -> int:
    import contextlib
    import io

    def check(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(f"self-test failed: {message}")

    check(normalize_symbols(" aapl, MSFT,,aapl ") == ["AAPL", "MSFT"], "symbol normalization")
    check(sanitize_csv_cell("=BAD") == "'=BAD", "CSV formula sanitization")
    check(commodity_probability(0.5) == NEUTRAL_COMMODITY_PROBABILITY, "neutral commodity probability stays neutral")
    check(
        len(monte_carlo_terminal_prices(100.0, 30, 123, 0.25, seed=11)) == 123,
        "Monte Carlo honors requested simulation count",
    )
    bs_price = black_scholes_price("put", 100.0, 95.0, 30 / CALENDAR_DAYS, 0.01, 0.25)
    check(math.isfinite(bs_price) and bs_price > 0, "Black-Scholes put price")
    binomial_price = american_option_binomial(100.0, 95.0, 30 / CALENDAR_DAYS, 0.01, 0.25, option_type="put")
    check(math.isfinite(binomial_price) and binomial_price > 0, "binomial put price")

    pd = require_module("pandas")
    expiration = (dt.date.today() + dt.timedelta(days=30)).strftime(DATE_FORMAT)
    chain = pd.DataFrame(
        [
            {"option_type": "put", "strike": 90, "bid": 0.15, "ask": 0.25, "greeks": {"mid_iv": 0.28}, "volume": 150, "open_interest": 900},
            {"option_type": "put", "strike": 95, "bid": 0.65, "ask": 0.80, "greeks": {"mid_iv": 0.27}, "volume": 120, "open_interest": 800},
            {"option_type": "put", "strike": 100, "bid": 2.20, "ask": 2.40, "greeks": {"mid_iv": 0.26}, "volume": 200, "open_interest": 1000},
        ]
    )
    noisy_chain = pd.DataFrame(
        [
            {"strike": "90", "bid": "0.10", "ask": "0.20", "greeks": {"mid_iv": "0.28"}, "volume": None, "open_interest": 5},
            {"strike": "90", "bid": "0.12", "ask": "0.22", "greeks": {"mid_iv": "0.27"}, "volume": 7, "open_interest": None},
            {"strike": "95", "bid": "bad", "ask": "0.80", "greeks": {"mid_iv": "0.27"}},
            {"strike": "100", "bid": "2.00", "ask": "1.90", "greeks": {"mid_iv": "0.26"}},
            {"strike": "105", "bid": "2.50", "ask": "2.75", "greeks": {}},
        ]
    )
    prepared = prepare_put_chain(noisy_chain)
    check(len(prepared) == 1 and float(prepared.iloc[0]["strike"]) == 90.0, "provider row filtering and dedupe")

    quote_payloads = [
        {"quotes": {"quote": {"last": "0", "mark": "101.25", "close": "100.50"}}},
        {"quotes": None},
    ]
    original_get_json = globals()["get_json"]
    try:
        globals()["get_json"] = lambda *args, **kwargs: quote_payloads.pop(0)
        check(get_stock_price("TEST", "token") == 101.25, "quote fallback uses first positive price")
        try:
            get_stock_price("BROKEN", "token")
        except RuntimeError as exc:
            check("quote" in str(exc).lower(), "malformed quote response has clear error")
        else:
            check(False, "malformed quote response should fail")
    finally:
        globals()["get_json"] = original_get_json

    previous_logger_disabled = logger.disabled
    logger.disabled = True
    try:
        spreads = process_bull_put_spread(
            "TEST",
            expiration,
            chain,
            underlying_price=105.0,
            min_ror=0.05,
            max_strike_dist=0.25,
            simulations=64,
            volatility=0.25,
            risk_free_rate=0.01,
            min_prob_success=0.0,
            commodity_prob=NEUTRAL_COMMODITY_PROBABILITY,
            seed=7,
        )
    finally:
        logger.disabled = previous_logger_disabled
    check(bool(spreads), "expected at least one spread")
    check(all(0 <= spread.composite_probability <= 1 for spread in spreads), "bounded composite probabilities")
    check(all(spread.max_loss > 0 and spread.credit > 0 for spread in spreads), "positive spread risk/reward")

    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "self_test.csv"
        write_results_csv(spreads[:1], output)
        text = output.read_text(encoding="utf-8")
        check("Symbol,Spread Type,Expiration" in text, "CSV header")
        check("TEST,bull_put" in text, "CSV row")

        original_fetch_option_chains = globals()["fetch_option_chains"]
        original_get_stock_price = globals()["get_stock_price"]
        original_get_historical_volatility = globals()["get_historical_volatility"]
        try:
            globals()["fetch_option_chains"] = lambda symbols, mindte, maxdte, api_token, batch_size: (
                [("GOOD", expiration, chain), ("BAD", expiration, chain)],
                [],
            )

            def fake_price(symbol: str, api_token: str) -> float:
                if symbol == "BAD":
                    raise RuntimeError("fake quote failure")
                return 105.0

            globals()["get_stock_price"] = fake_price
            globals()["get_historical_volatility"] = lambda symbol, api_token, cache: 0.25
            scan_output = Path(temp_dir) / "scan.csv"
            previous_logger_disabled = logger.disabled
            logger.disabled = True
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    scan_results = run_scan(
                        ScanConfig(
                            symbols=["GOOD", "BAD"],
                            mindte=1,
                            maxdte=60,
                            top_n=5,
                            min_ror=0.05,
                            max_strike_dist=0.25,
                            output=scan_output,
                            batch_size=2,
                            simulations=64,
                            risk_free_rate=0.01,
                            min_prob_success=0.0,
                            api_token="token",
                            commodities_api_key=None,
                            use_commodity_score=False,
                            max_commodities=1,
                        )
                    )
            finally:
                logger.disabled = previous_logger_disabled
            check(scan_results and all(spread.symbol == "GOOD" for spread in scan_results), "partial scan keeps good symbol")
            check("GOOD,bull_put" in scan_output.read_text(encoding="utf-8"), "partial scan writes CSV")
        finally:
            globals()["fetch_option_chains"] = original_fetch_option_chains
            globals()["get_stock_price"] = original_get_stock_price
            globals()["get_historical_volatility"] = original_get_historical_volatility

    print("Self-test passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args)
    try:
        validate_args(args, parser)
        if args.self_test:
            return run_self_test()
        run_scan(config_from_args(args))
        return 0
    except RuntimeError as exc:
        if args.debug:
            raise
        parser.exit(1, f"error: {exc}\n")
    except OSError as exc:
        if args.debug:
            raise
        parser.exit(1, f"error: file operation failed: {exc}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
