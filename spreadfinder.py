import argparse
import datetime
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm, t
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import islice
import os
import logging
from scipy.stats.qmc import Sobol
from arch import arch_model  # For GARCH
import math

TRADIER_API_URL = "https://api.tradier.com/v1"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    options = data['options']['option']
    puts = [opt for opt in options if opt['option_type'] == 'put']
    calls = [opt for opt in options if opt['option_type'] == 'call']
    return puts, calls

def get_historical_volatility(symbol, api_token, days=252):
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
    
    # Using GARCH(1,1) model for volatility estimation
    # Rescale log_returns for better GARCH model performance
    log_returns *= 10  # Rescaling
    model = arch_model(log_returns, vol='Garch', p=1, q=1, rescale=False)
    res = model.fit(disp="off")
    forecast = res.forecast(horizon=1)
    annualized_volatility = np.sqrt(forecast.variance.values[-1, :][0]) * np.sqrt(days)
    
    return annualized_volatility

def heston_simulation(S0, T, r, kappa, theta, sigma, rho, v0, steps, simulations):
    dt = T / steps
    prices = np.zeros((steps + 1, simulations))
    volatilities = np.zeros((steps + 1, simulations))
    
    prices[0] = S0
    volatilities[0] = v0

    for t in range(1, steps + 1):
        z1 = np.random.normal(size=simulations)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=simulations)
        
        volatilities[t] = (volatilities[t-1] + kappa * (theta - np.maximum(volatilities[t-1], 0)) * dt 
                           + sigma * np.sqrt(np.maximum(volatilities[t-1], 0)) * np.sqrt(dt) * z1)
        
        prices[t] = prices[t-1] * np.exp((r - 0.5 * volatilities[t-1]) * dt + np.sqrt(np.maximum(volatilities[t-1], 0)) * np.sqrt(dt) * z2)
    
    return prices

def get_stock_data(symbol, mindte, maxdte, api_token):
    logger.info(f"Fetching stock data for {symbol} with min DTE {mindte} and max DTE {maxdte}")
    expirations = get_option_expirations(symbol, api_token)

    valid_exp = []
    for exp in expirations:
        exp_date = datetime.datetime.strptime(exp, '%Y-%m-%d')
        dte = (exp_date - datetime.datetime.now()).days
        if mindte <= dte <= maxdte:
            valid_exp.append(exp)

    puts = []
    calls = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_option_chain, symbol, exp, api_token): exp for exp in valid_exp}
        for future in as_completed(futures):
            exp = futures[future]
            opt_puts_filtered, opt_calls_filtered = future.result()
            puts.append((exp, pd.DataFrame(opt_puts_filtered)))
            calls.append((exp, pd.DataFrame(opt_calls_filtered)))

    logger.info(f"Found {len(puts)} put chains and {len(calls)} call chains for {symbol} with min DTE {mindte} and max DTE {maxdte}")
    return puts, calls

def monte_carlo_simulation(current_price, days, simulations, volatility, use_t_dist=False, df=3):
    # Ensure the number of simulations is a power of two
    simulations = 2**int(np.ceil(np.log2(simulations)))

    dt = 1 / 252  # daily time step

    if use_t_dist:
        z = t.rvs(df, size=(simulations, days))
    else:
        z = np.random.normal(size=(simulations, days))
    
    log_returns = (volatility * np.sqrt(dt)) * z
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

def process_bull_put_spread(exp, put_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate):
    def process_single_spread(i, j):
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

            time_to_expiration = (datetime.datetime.strptime(exp, '%Y-%m-%d') - datetime.datetime.now()).days / 365.0
            iv = short_put['greeks']['mid_iv']
            if iv is None or iv <= 0:
                return []

            theoretical_price_binomial = american_option_binomial(underlying_price, short_put['strike'], time_to_expiration, risk_free_rate, iv, option_type='put')
            theoretical_price_bs = black_scholes_price('put', underlying_price, short_put['strike'], time_to_expiration, risk_free_rate, iv)

            pricing_state = "fairly priced"
            if short_put['bid'] < theoretical_price_binomial and short_put['bid'] < theoretical_price_bs:
                pricing_state = "underpriced"
            elif short_put['bid'] > theoretical_price_binomial and short_put['bid'] > theoretical_price_bs:
                pricing_state = "overpriced"

            d1 = (np.log(underlying_price / short_put['strike']) + (0.5 * iv**2) * time_to_expiration) / (iv * np.sqrt(time_to_expiration))
            probability_of_success = norm.cdf(d1)

            # Monte Carlo Simulation
            mc_prices = monte_carlo_simulation(underlying_price, int(time_to_expiration * 252), simulations, iv, use_t_dist=True)
            mc_prob_profit = np.mean(mc_prices[-1] > short_put['strike'])

            return [(exp, short_put['strike'], long_put['strike'], credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit, pricing_state)]
        return []

    results = []
    put_chain = put_chain.sort_values(by='strike')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_spread, i, j) for i in range(len(put_chain)) for j in range(i + 1, len(put_chain))]
        total_spreads = len(futures)
        completed_spreads = 0

        for future in as_completed(futures):
            results.extend(future.result())
            completed_spreads += 1
            if completed_spreads % 100 == 0 or completed_spreads == total_spreads:
                logger.info(f"Processed {completed_spreads}/{total_spreads} bull put spreads")

    return results

def batch_futures(iterator, batch_size):
    """Helper function to batch futures."""
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def find_bull_put_spreads(puts, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility, risk_free_rate):
    logger.info(f"Finding bull put spreads for underlying price {underlying_price} using batch size {batch_size}")
    bull_put_spreads = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures_iterator = (
            executor.submit(process_bull_put_spread, exp, put_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate)
            for exp, put_chain in puts
        )
        batches_processed = 0
        for batch in batch_futures(futures_iterator, batch_size):
            for future in as_completed(batch):
                bull_put_spreads.extend(future.result())
            batches_processed += 1
            logger.info(f"Processed batch {batches_processed}")

    return bull_put_spreads

def process_iron_condor(exp, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate):
    def process_single_condor(i, j, k, l):
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

            time_to_expiration = (datetime.datetime.strptime(exp, '%Y-%m-%d') - datetime.datetime.now()).days / 365.0
            iv_put = short_put['greeks']['mid_iv']
            iv_call = short_call['greeks']['mid_iv']
            if iv_put is None or iv_put <= 0 or iv_call is None or iv_call <= 0:
                return []

            theoretical_price_put_binomial = american_option_binomial(underlying_price, short_put['strike'], time_to_expiration, risk_free_rate, iv_put, option_type='put')
            theoretical_price_put_bs = black_scholes_price('put', underlying_price, short_put['strike'], time_to_expiration, risk_free_rate, iv_put)

            theoretical_price_call_binomial = american_option_binomial(underlying_price, short_call['strike'], time_to_expiration, risk_free_rate, iv_call, option_type='call')
            theoretical_price_call_bs = black_scholes_price('call', underlying_price, short_call['strike'], time_to_expiration, risk_free_rate, iv_call)

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

            d1_put = (np.log(underlying_price / short_put['strike']) + (0.5 * iv_put**2) * time_to_expiration) / (iv_put * np.sqrt(time_to_expiration))
            d1_call = (np.log(underlying_price / short_call['strike']) + (0.5 * iv_call**2) * time_to_expiration) / (iv_call * np.sqrt(time_to_expiration))
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

    results = []
    put_chain = put_chain.sort_values(by='strike')
    call_chain = call_chain.sort_values(by='strike')

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_condor, i, j, k, l)
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

def find_best_spreads(puts, calls, underlying_price, top_n, min_ror, max_strike_dist, batch_size, simulations, volatility, include_iron_condors, risk_free_rate):
    bull_put_spreads = find_bull_put_spreads(puts, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility, risk_free_rate)
    combined_spreads = []

    for spread in bull_put_spreads:
        combined_spreads.append((*spread, 'bull_put'))

    if include_iron_condors:
        iron_condors = find_iron_condors(puts, calls, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility, risk_free_rate)
        for spread in iron_condors:
            combined_spreads.append((*spread, 'iron_condor'))

    combined_spreads.sort(key=lambda x: (x[-3], x[-2]), reverse=True)

    spread_data = []
    for spread in combined_spreads:
        if spread[-1] == 'bull_put':
            exp, short_strike, long_strike, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit, pricing_state, spread_type = spread
            spread_data.append([spread_type, exp, short_strike, long_strike, None, None, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit, pricing_state, None])
        elif spread[-1] == 'iron_condor':
            exp, short_put_strike, long_put_strike, short_call_strike, long_call_strike, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit, pricing_state_put, pricing_state_call, spread_type = spread
            spread_data.append([spread_type, exp, short_put_strike, long_put_strike, short_call_strike, long_call_strike, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit, pricing_state_put, pricing_state_call])

    return spread_data[:top_n]

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

    args = argparser.parse_args()
    
    underlying_price = get_stock_price(args.symbol, args.api_token)
    volatility = get_historical_volatility(args.symbol, args.api_token)
    puts, calls = get_stock_data(args.symbol, args.mindte, args.maxdte, args.api_token)
    
    logger.info(f"Underlying price: {underlying_price}")
    logger.info(f"Historical volatility: {volatility}")

    best_spreads = find_best_spreads(puts, calls, underlying_price, args.top_n, args.min_ror, args.max_strike_dist, args.batch_size, args.simulations, volatility, args.include_iron_condors, args.risk_free_rate)
    
    logger.info("Best Spreads:")
    df = pd.DataFrame(best_spreads, columns=['Type', 'Expiration', 'Short Put Strike', 'Long Put Strike', 'Short Call Strike', 'Long Call Strike', 'Credit', 'Max Loss', 'Return on Risk', 'Probability of Success', 'MC Probability of Profit', 'Pricing State (Put)', 'Pricing State (Call)'])
    df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

    for spread in best_spreads:
        if spread[0] == 'bull_put':
            logger.info(f"Bull Put Spread:\n\tExpiration: {spread[1]}\n\tShort Strike: {spread[2]}\n\tLong Strike: {spread[3]}\n\tCredit: {spread[6]}\n\tMax Loss: {spread[7]}\n\tReturn on Risk: {spread[8]*100:.2f}%\n\tProbability of Success: {spread[9]*100:.2f}%\n\tMC Probability of Profit: {spread[10]*100:.2f}%\n\tPricing State: {spread[11]}")
        elif spread[0] == 'iron_condor':
            logger.info(f"Iron Condor:\n\tExpiration: {spread[1]}\n\tShort Put Strike: {spread[2]}\n\tLong Put Strike: {spread[3]}\n\tShort Call Strike: {spread[4]}\n\tLong Call Strike: {spread[5]}\n\tCredit: {spread[6]}\n\tMax Loss: {spread[7]}\n\tReturn on Risk: {spread[8]*100:.2f}%\n\tProbability of Success: {spread[9]*100:.2f}%\n\tMC Probability of Profit: {spread[10]*100:.2f}%\n\tPricing State (Put): {spread[11]}\n\tPricing State (Call): {spread[12]}")

    endTime = datetime.datetime.now()
    logger.info(f"Time taken: {endTime - startTime}")
