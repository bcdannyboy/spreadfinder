import argparse
import datetime
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import logging
from itertools import islice
from arch import arch_model  # For GARCH
from ratelimit import limits, sleep_and_retry
from multiprocessing import Manager
import time
import shelve  # Simple persistent storage for caching

TRADIER_API_URL = "https://api.tradier.com/v1"
RATE_LIMIT = 60  # Number of requests per minute
CACHE_FILE = 'volatility_cache.db'  # Cache file for storing volatility data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@sleep_and_retry
@limits(calls=RATE_LIMIT, period=60)
def get_stock_price(symbol, api_token):
    url = f"{TRADIER_API_URL}/markets/quotes"
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Accept': 'application/json'
    }
    params = {'symbols': symbol}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    return data['quotes']['quote']['last']

@sleep_and_retry
@limits(calls=RATE_LIMIT, period=60)
def get_option_expirations(symbol, api_token):
    url = f"{TRADIER_API_URL}/markets/options/expirations"
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Accept': 'application/json'
    }
    params = {'symbol': symbol}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    return data['expirations']['date']

@sleep_and_retry
@limits(calls=RATE_LIMIT, period=60)
def get_option_chain(symbol, expiration, api_token):
    url = f"{TRADIER_API_URL}/markets/options/chains"
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Accept': 'application/json'
    }
    params = {'symbol': symbol, 'expiration': expiration, 'greeks': 'true'}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    if 'options' in data and 'option' in data['options']:
        options = data['options']['option']
        puts = [opt for opt in options if opt['option_type'] == 'put']
        calls = [opt for opt in options if opt['option_type'] == 'call']
        return puts, calls
    else:
        logger.warning(f"No option data found for {symbol} with expiration {expiration}")
        return [], []

@sleep_and_retry
@limits(calls=RATE_LIMIT, period=60)
def get_historical_volatility(symbol, api_token, cache, days=252):
    cache_key = f"{symbol}_{days}"
    if cache_key in cache:
        return cache[cache_key]

    url = f"{TRADIER_API_URL}/markets/history"
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Accept': 'application/json'
    }
    params = {'symbol': symbol, 'interval': 'daily', 'start': (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d'), 'end': datetime.datetime.now().strftime('%Y-%m-%d')}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    historical_prices = [day['close'] for day in data['history']['day']]
    
    log_returns = np.log(np.array(historical_prices[1:]) / np.array(historical_prices[:-1]))
    
    # Rescaling log returns by a factor of 100
    log_returns *= 100
    
    # Using GARCH(1,1) model for volatility estimation
    model = arch_model(log_returns, vol='Garch', p=1, q=1, rescale=False)
    res = model.fit(disp="off")
    forecast = res.forecast(horizon=1)
    annualized_volatility = np.sqrt(forecast.variance.values[-1, :][0]) * np.sqrt(days)

    cache[cache_key] = annualized_volatility

    return annualized_volatility

def get_historical_volatility_adjusted(symbol, api_token, dte, cache, days=252):
    annualized_volatility = get_historical_volatility(symbol, api_token, cache, days)
    adjusted_volatility = annualized_volatility * np.sqrt(dte / 365.0)
    return adjusted_volatility

def monte_carlo_simulation(current_price, days, simulations, volatility, use_t_dist=False, df=3, log_return_cap=10):
    # Ensure the number of simulations is a power of two
    simulations = 2**int(np.ceil(np.log2(simulations)))

    dt = 1 / 252  # daily time step

    if use_t_dist:
        z = t.rvs(df, size=(simulations, days))
    else:
        z = np.random.normal(size=(simulations, days))
    
    log_returns = (volatility * np.sqrt(dt)) * z
    
    # Cap log returns to prevent overflow
    log_returns = np.clip(log_returns, -log_return_cap, log_return_cap)
    
    price_paths = current_price * np.exp(np.cumsum(log_returns, axis=1))
    price_paths = np.hstack([np.full((simulations, 1), current_price), price_paths])
    
    return price_paths.T

def black_scholes_price(option_type, S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def american_option_binomial(S, K, T, r, sigma, option_type='call', steps=100):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    asset_prices = np.zeros((steps + 1, steps + 1))
    option_values = np.zeros((steps + 1, steps + 1))
    
    for i in range(steps + 1):
        for j in range(i + 1):
            asset_prices[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # Initialize option values at maturity
    if option_type == 'call':
        option_values[:, steps] = np.maximum(0, asset_prices[:, steps] - K)
    elif option_type == 'put':
        option_values[:, steps] = np.maximum(0, K - asset_prices[:, steps])
    
    # Backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold_value = np.exp(-r * dt) * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])
            if option_type == 'call':
                exercise_value = np.maximum(0, asset_prices[j, i] - K)
            elif option_type == 'put':
                exercise_value = np.maximum(0, K - asset_prices[j, i])
            option_values[j, i] = np.maximum(hold_value, exercise_value)
    
    return option_values[0, 0]

def get_stock_data(symbol, mindte, maxdte, api_token):
    logger.info(f"Fetching stock data for {symbol} with min DTE {mindte} and max DTE {maxdte}")
    expirations = get_option_expirations(symbol, api_token)

    valid_exp = []
    for exp in expirations:
        try:
            exp_date = datetime.datetime.strptime(exp, '%Y-%m-%d')
            dte = (exp_date - datetime.datetime.now()).days
            if mindte <= dte <= maxdte:
                valid_exp.append(exp)
        except ValueError as e:
            logger.error(f"Error parsing expiration date: {exp}, error: {e}")

    puts = []
    calls = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_option_chain, symbol, exp, api_token): exp for exp in valid_exp}
        for future in as_completed(futures):
            exp = futures[future]
            opt_puts_filtered, opt_calls_filtered = future.result()
            if opt_puts_filtered or opt_calls_filtered:
                puts.append((exp, pd.DataFrame(opt_puts_filtered)))
                calls.append((exp, pd.DataFrame(opt_calls_filtered)))
            else:
                logger.warning(f"No options data found for expiration {exp}")

    logger.info(f"Found {len(puts)} put chains and {len(calls)} call chains for {symbol} with min DTE {mindte} and max DTE {maxdte}")
    return puts, calls

def process_single_spread(i, j, put_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, api_token, symbol, exp, cache):
    short_put = put_chain.iloc[j]
    long_put = put_chain.iloc[i]

    if short_put['strike'] > underlying_price * (1 + max_strike_dist) or short_put['strike'] < underlying_price * (1 - max_strike_dist):
        return []

    if short_put['strike'] > long_put['strike']:
        credit = short_put['bid'] - long_put['ask']
        max_loss = (short_put['strike'] - long_put['strike']) - credit
        if max_loss <= 0:
            return []

        return_on_risk = credit / max_loss

        if return_on_risk < min_ror:
            return []

        time_to_expiration = (datetime.datetime.strptime(exp, '%Y-%m-%d') - datetime.datetime.now()).days
        iv = short_put['greeks']['mid_iv']
        if iv is None or iv <= 0:
            return []

        theoretical_price_binomial = american_option_binomial(underlying_price, short_put['strike'], time_to_expiration / 365.0, risk_free_rate, iv, option_type='put')
        theoretical_price_bs = black_scholes_price('put', underlying_price, short_put['strike'], time_to_expiration / 365.0, risk_free_rate, iv)

        pricing_state = "fairly priced"
        if short_put['bid'] < theoretical_price_binomial and short_put['bid'] < theoretical_price_bs:
            pricing_state = "underpriced"
        elif short_put['bid'] > theoretical_price_binomial and short_put['bid'] > theoretical_price_bs:
            pricing_state = "overpriced"

        d1 = (np.log(underlying_price / short_put['strike']) + (0.5 * iv**2) * (time_to_expiration / 365.0)) / (iv * np.sqrt(time_to_expiration / 365.0))
        probability_of_success = norm.cdf(d1)

        # Monte Carlo Simulation with and without DTE adjustment
        mc_prices_no_dte = monte_carlo_simulation(underlying_price, int(time_to_expiration), simulations, volatility, use_t_dist=True)
        mc_prob_profit_no_dte = np.mean(mc_prices_no_dte[-1] > short_put['strike']) if len(mc_prices_no_dte) > 1 else 0
        
        adjusted_volatility = get_historical_volatility_adjusted(symbol, api_token, time_to_expiration, cache)
        mc_prices_with_dte = monte_carlo_simulation(underlying_price, int(time_to_expiration), simulations, adjusted_volatility, use_t_dist=True)
        mc_prob_profit_with_dte = np.mean(mc_prices_with_dte[-1] > short_put['strike']) if len(mc_prices_with_dte) > 1 else 0

        return [(exp, short_put['strike'], long_put['strike'], credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, pricing_state)]
    return []

def process_bull_put_spread(exp, put_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, api_token, symbol, cache):
    results = []
    put_chain = put_chain.sort_values(by='strike')
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_spread, i, j, put_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, api_token, symbol, exp, cache) for i in range(len(put_chain)) for j in range(i + 1, len(put_chain))]
        total_spreads = len(futures)
        completed_spreads = 0

        for future in as_completed(futures):
            results.extend(future.result())
            completed_spreads += 1
            if completed_spreads % 100 == 0 or completed_spreads == total_spreads:
                logger.info(f"Processed {completed_spreads}/{total_spreads} bull put spreads")

    return results

def process_single_condor(i, j, k, l, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, exp):
    short_put = put_chain.iloc[j]
    long_put = put_chain.iloc[i]
    short_call = call_chain.iloc[k]
    long_call = call_chain.iloc[l]

    if (short_put['strike'] > underlying_price * (1 + max_strike_dist) or short_put['strike'] < underlying_price * (1 - max_strike_dist) or
        short_call['strike'] > underlying_price * (1 + max_strike_dist) or short_call['strike'] < underlying_price * (1 - max_strike_dist)):
        return []

    if short_put['strike'] > long_put['strike'] and short_call['strike'] < long_call['strike']:
        credit = short_put['bid'] - long_put['ask'] + short_call['bid'] - long_call['ask']
        max_loss_put = (short_put['strike'] - long_put['strike']) - credit
        max_loss_call = (long_call['strike'] - short_call['strike']) - credit
        max_loss = max(max_loss_put, max_loss_call)
        if max_loss <= 0:
            return []

        return_on_risk = credit / max_loss

        if return_on_risk < min_ror:
            return []

        time_to_expiration = (datetime.datetime.strptime(exp, '%Y-%m-%d') - datetime.datetime.now()).days
        iv_put = short_put['greeks']['mid_iv']
        iv_call = short_call['greeks']['mid_iv']
        if iv_put is None or iv_put <= 0 or iv_call is None or iv_call <= 0:
            return []

        theoretical_price_put_binomial = american_option_binomial(underlying_price, short_put['strike'], time_to_expiration / 365.0, risk_free_rate, iv_put, option_type='put')
        theoretical_price_put_bs = black_scholes_price('put', underlying_price, short_put['strike'], time_to_expiration / 365.0, risk_free_rate, iv_put)

        theoretical_price_call_binomial = american_option_binomial(underlying_price, short_call['strike'], time_to_expiration / 365.0, risk_free_rate, iv_call, option_type='call')
        theoretical_price_call_bs = black_scholes_price('call', underlying_price, short_call['strike'], time_to_expiration / 365.0, risk_free_rate, iv_call)

        pricing_state_put = "fairly priced"
        if short_put['bid'] < theoretical_price_put_binomial and short_put['bid'] < theoretical_price_put_bs:
            pricing_state_put = "underpriced"
        elif short_put['bid'] > theoretical_price_put_binomial and short_put['bid'] > theoretical_price_put_bs:
            pricing_state_put = "overpriced"

        pricing_state_call = "fairly priced"
        if short_call['bid'] < theoretical_price_call_binomial and short_call['bid'] < theoretical_price_call_bs:
            pricing_state_call = "underpriced"
        elif short_call['bid'] > theoretical_price_call_binomial and short_call['bid'] > theoretical_price_call_bs:
            pricing_state_call = "overpriced"

        d1_put = (np.log(underlying_price / short_put['strike']) + (0.5 * iv_put**2) * (time_to_expiration / 365.0)) / (iv_put * np.sqrt(time_to_expiration / 365.0))
        d1_call = (np.log(underlying_price / short_call['strike']) + (0.5 * iv_call**2) * (time_to_expiration / 365.0)) / (iv_call * np.sqrt(time_to_expiration / 365.0))
        probability_of_success_put = norm.cdf(d1_put)
        probability_of_success_call = 1 - norm.cdf(d1_call)
        probability_of_success = (probability_of_success_put + probability_of_success_call) / 2

        # Monte Carlo Simulation
        mc_prices = monte_carlo_simulation(underlying_price, int(time_to_expiration * 252), simulations, iv_put, use_t_dist=True)
        mc_prob_profit_put = np.mean(mc_prices[-1] > short_put['strike'])
        mc_prob_profit_call = np.mean(mc_prices[-1] < short_call['strike'])
        mc_prob_profit = (mc_prob_profit_put + mc_prob_profit_call) / 2

        return [(exp, short_put['strike'], long_put['strike'], short_call['strike'], long_call['strike'], credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit, pricing_state_put, pricing_state_call)]
    return []

def process_iron_condor(exp, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate):
    results = []
    put_chain = put_chain.sort_values(by='strike')
    call_chain = call_chain.sort_values(by='strike')

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_condor, i, j, k, l, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, exp)
                   for i in range(len(put_chain)) for j in range(i + 1, len(put_chain))
                   for k in range(len(call_chain)) for l in range(k + 1, len(call_chain))]
        total_condors = len(futures)
        completed_condors = 0

        for future in as_completed(futures):
            results.extend(future.result())
            completed_condors += 1
            if completed_condors % 100 == 0 or completed_condors == total_condors:
                logger.info(f"Processed {completed_condors}/{total_condors} iron condors")

    return results

def batch_futures(iterator, batch_size):
    """Helper function to batch futures."""
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def find_bull_put_spreads(puts, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility, risk_free_rate, api_token, symbol, cache):
    logger.info(f"Finding bull put spreads for underlying price {underlying_price} using batch size {batch_size}")
    bull_put_spreads = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures_iterator = (
            executor.submit(process_bull_put_spread, exp, put_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, api_token, symbol, cache)
            for exp, put_chain in puts
        )
        batches_processed = 0
        for batch in batch_futures(futures_iterator, batch_size):
            for future in as_completed(batch):
                bull_put_spreads.extend(future.result())
            batches_processed += 1
            logger.info(f"Processed batch {batches_processed}")

    return bull_put_spreads

def find_iron_condors(puts, calls, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility, risk_free_rate):
    logger.info(f"Finding iron condors for underlying price {underlying_price} using batch size {batch_size}")
    iron_condors = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures_iterator = (
            executor.submit(process_iron_condor, exp, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate)
            for (exp, put_chain), (exp_c, call_chain) in zip(puts, calls)
            if exp == exp_c
        )
        batches_processed = 0
        for batch in batch_futures(futures_iterator, batch_size):
            for future in as_completed(batch):
                iron_condors.extend(future.result())
            batches_processed += 1
            logger.info(f"Processed batch {batches_processed}")

    return iron_condors

def find_best_spreads(puts, calls, underlying_price, top_n, min_ror, max_strike_dist, batch_size, simulations, volatility, include_iron_condors, risk_free_rate, api_token, symbol):
    with Manager() as manager:
        cache = manager.dict()

        bull_put_spreads = find_bull_put_spreads(puts, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility, risk_free_rate, api_token, symbol, cache)
        combined_spreads = []

        for spread in bull_put_spreads:
            combined_spreads.append(list((*spread, 'bull_put')))  # Convert tuple to list

        if include_iron_condors:
            iron_condors = find_iron_condors(puts, calls, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility, risk_free_rate, cache)
            for spread in iron_condors:
                combined_spreads.append(list((*spread, 'iron_condor')))  # Convert tuple to list

        # Filter out spreads that don't meet the minimum return on risk
        filtered_spreads = [spread for spread in combined_spreads if spread[5] >= min_ror]

        # Sort by probability of success, prioritizing the highest probabilities
        filtered_spreads.sort(key=lambda x: x[6], reverse=True)

        spread_data = []
        for spread in filtered_spreads[:top_n]:
            if spread[-1] == 'bull_put':
                exp, short_strike, long_strike, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, pricing_state, spread_type = spread
                average_probability = (probability_of_success + mc_prob_profit_no_dte + mc_prob_profit_with_dte) / 3
                spread_data.append([spread_type, exp, short_strike, long_strike, None, None, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, pricing_state, None, average_probability])
            elif spread[-1] == 'iron_condor':
                exp, short_put_strike, long_put_strike, short_call_strike, long_call_strike, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, pricing_state_put, pricing_state_call, spread_type = spread
                average_probability = (probability_of_success + mc_prob_profit_no_dte + mc_prob_profit_with_dte) / 3
                spread_data.append([spread_type, exp, short_put_strike, long_put_strike, short_call_strike, long_call_strike, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, pricing_state_put, pricing_state_call, average_probability])

        return spread_data

def plot_probabilities(spreads):
    bull_puts = [spread for spread in spreads if spread[0] == 'bull_put']
    iron_condors = [spread for spread in spreads if spread[0] == 'iron_condor']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    if bull_puts:
        exp_dates, no_dte_probs, with_dte_probs = zip(*[(spread[1], spread[10], spread[11]) for spread in bull_puts])
        axes[0].plot(exp_dates, no_dte_probs, label='No DTE Adjustment')
        axes[0].plot(exp_dates, with_dte_probs, label='With DTE Adjustment')
        axes[0].set_title('Bull Put Spreads')
        axes[0].legend()

    if iron_condors:
        exp_dates, no_dte_probs, with_dte_probs = zip(*[(spread[1], spread[10], spread[11]) for spread in iron_condors])
        axes[1].plot(exp_dates, no_dte_probs, label='No DTE Adjustment')
        axes[1].plot(exp_dates, with_dte_probs, label='With DTE Adjustment')
        axes[1].set_title('Iron Condors')
        axes[1].legend()

    plt.xlabel('Expiration Dates')
    plt.ylabel('Probability of Profit')
    plt.show()

if __name__ == '__main__':
    startTime = datetime.datetime.now()
    argparser = argparse.ArgumentParser(description="Find and rank option spreads")
    
    argparser.add_argument('-symbol', '-s', type=str, required=True, help='Stock symbol to fetch data for')
    argparser.add_argument('-mindte', '-m', type=int, required=True, help='Minimum days to expiration')
    argparser.add_argument('-maxdte', '-l', type=int, required=True, help='Maximum days to expiration')
    argparser.add_argument('-top_n', '-n', type=int, default=10, help='Number of top spreads to display')
    argparser.add_argument('-min_ror', '-r', type=float, default=0.15, help='Minimum return on risk')
    argparser.add_argument('-max_strike_dist', '-d', type=float, default=0.2, help='Maximum strike distance percentage')
    argparser.add_argument('-output', '-o', type=str, default='best_spreads.csv', help='Output CSV file path')
    argparser.add_argument('-batch_size', '-b', type=int, default=10, help='Batch size for processing tasks')
    argparser.add_argument('--include_iron_condors', action='store_true', help='Include iron condors in the search')
    argparser.add_argument('-api_token', type=str, required=True, help='Tradier API token')
    argparser.add_argument('-simulations', type=int, default=1000, help='Number of Monte Carlo simulations')
    argparser.add_argument('-risk_free_rate', '-rf', type=float, default=0.01, help='Risk-free interest rate')
    argparser.add_argument('--use_heston', action='store_true', help='Use Heston model for simulations')
    argparser.add_argument('--plot', action='store_true', help='Show probability of profit plot')
    args = argparser.parse_args()
    
    with Manager() as manager:
        cache = manager.dict()
        
        underlying_price = get_stock_price(args.symbol, args.api_token)
        volatility = get_historical_volatility(args.symbol, args.api_token, cache)
        puts, calls = get_stock_data(args.symbol, args.mindte, args.maxdte, args.api_token)
        
        logger.info(f"Underlying price: {underlying_price}")
        logger.info(f"Historical volatility: {volatility}")

        best_spreads = find_best_spreads(puts, calls, underlying_price, args.top_n, args.min_ror, args.max_strike_dist, args.batch_size, args.simulations, volatility, args.include_iron_condors, args.risk_free_rate, args.api_token, args.symbol)

        logger.info(f"Best Spreads for {args.symbol} with min DTE {args.mindte} and max DTE {args.maxdte}")
        df = pd.DataFrame(best_spreads, columns=['Type', 'Expiration', 'Short Put Strike', 'Long Put Strike', 'Short Call Strike', 'Long Call Strike', 'Credit', 'Max Loss', 'Return on Risk', 'Probability of Success', 'MC Probability of Profit (No DTE)', 'MC Probability of Profit (With DTE)', 'Pricing State (Put)', 'Pricing State (Call)', 'Average Probability'])
        df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")

        for spread in best_spreads:
            if spread[0] == 'bull_put':
                logger.info(f"Bull Put Spread:\n\tExpiration: {spread[1]}\n\tShort Strike: {spread[2]}\n\tLong Strike: {spread[3]}\n\tCredit: {spread[6]}\n\tMax Loss: {spread[7]}\n\tReturn on Risk: {spread[8]*100:.2f}%\n\tProbability of Success: {spread[9]*100:.2f}%\n\tMC Probability of Profit (No DTE): {spread[10]*100:.2f}%\n\tMC Probability of Profit (With DTE): {spread[11]*100:.2f}%\n\tPricing State: {spread[12]}")
            elif spread[0] == 'iron_condor':
                logger.info(f"Iron Condor:\n\tExpiration: {spread[1]}\n\tShort Put Strike: {spread[2]}\n\tLong Put Strike: {spread[3]}\n\tShort Call Strike: {spread[4]}\n\tLong Call Strike: {spread[5]}\n\tCredit: {spread[6]}\n\tMax Loss: {spread[7]}\n\tReturn on Risk: {spread[8]*100:.2f}%\n\tProbability of Success: {spread[9]*100:.2f}%\n\tMC Probability of Profit (No DTE): {spread[10]*100:.2f}%\n\tMC Probability of Profit (With DTE): {spread[11]*100:.2f}%\n\tPricing State (Put): {spread[12]}\n\tPricing State (Call): {spread[13]}")

        if args.plot:
            plot_probabilities(best_spreads)

        endTime = datetime.datetime.now()
        logger.info(f"Time taken: {endTime - startTime}")
