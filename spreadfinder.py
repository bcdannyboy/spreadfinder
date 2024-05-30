import argparse
import datetime
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.stats import norm, t, qmc
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import logging
from itertools import islice
from arch import arch_model  # For GARCH
from ratelimit import limits, sleep_and_retry
from multiprocessing import Manager
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

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
    annualized_volatility = np.minimum(annualized_volatility, 2.0)  # Bound the annualized volatility

    cache[cache_key] = annualized_volatility

    return annualized_volatility

def get_historical_volatility_adjusted(symbol, api_token, dte, cache, days=252):
    annualized_volatility = get_historical_volatility(symbol, api_token, cache, days)
    adjusted_volatility = annualized_volatility * np.sqrt(dte / 365.0)
    return adjusted_volatility

def monte_carlo_simulation(current_price, days, simulations, volatility, use_t_dist=False, df=3, log_return_cap=None, use_heston=False, kappa=2.0, theta=0.02, xi=0.1, rho=-0.7):
    from scipy.stats import t
    dt = 1 / 252  # daily time step
    simulations = 2**int(np.ceil(np.log2(simulations)))  # Ensure power of 2

    # Generate samples using Latin Hypercube Sampling
    lhs_sampler = qmc.LatinHypercube(d=days)
    lhs_samples = lhs_sampler.random(n=simulations)
    if use_t_dist:
        lhs_samples = t.ppf(lhs_samples, df)
    else:
        lhs_samples = norm.ppf(lhs_samples)

    if use_heston:
        # Vectorized Heston model implementation
        Vt = np.full(simulations, volatility**2)
        price_paths = np.zeros((days + 1, simulations))
        price_paths[0] = current_price
        Z1 = lhs_samples.T
        Z2 = np.random.normal(size=(days, simulations))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        for t in range(1, days + 1):
            Vt = np.maximum(0, Vt + kappa * (theta - Vt) * dt + xi * np.sqrt(Vt) * np.sqrt(dt) * Z1[t-1])
            Vt = np.minimum(Vt, 4 * volatility**2)  # Bound volatility
            price_paths[t] = price_paths[t-1] * np.exp((Vt - 0.5 * Vt) * dt + np.sqrt(Vt * dt) * Z2[t-1])

        final_prices = price_paths[-1]
        return price_paths

    log_returns = (volatility * np.sqrt(dt)) * lhs_samples
    log_returns = np.clip(log_returns, -log_return_cap, log_return_cap) if log_return_cap else log_returns

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

def get_stock_data(symbols, mindte, maxdte, api_token):
    all_puts = []
    all_calls = []
    
    for symbol in symbols:
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

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(get_option_chain, symbol, exp, api_token): exp for exp in valid_exp}
            for future in as_completed(futures):
                exp = futures[future]
                opt_puts_filtered, opt_calls_filtered = future.result()
                if opt_puts_filtered or opt_calls_filtered:
                    all_puts.append((symbol, exp, pd.DataFrame(opt_puts_filtered)))
                    all_calls.append((symbol, exp, pd.DataFrame(opt_calls_filtered)))
                else:
                    logger.warning(f"No options data found for expiration {exp}")

        logger.info(f"Found {len(all_puts)} put chains and {len(all_calls)} call chains for {symbol} with min DTE {mindte} and max DTE {maxdte}")
    
    return all_puts, all_calls

def process_single_spread(i, j, put_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, api_token, symbol, exp, cache, min_prob_success):
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

        # Ensure 'greeks' key exists and 'mid_iv' is not None
        if 'greeks' not in short_put or short_put['greeks'] is None or 'mid_iv' not in short_put['greeks'] or short_put['greeks']['mid_iv'] is None:
            return []

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

        # Heston Model Simulations
        mc_prices_heston = monte_carlo_simulation(underlying_price, int(time_to_expiration), simulations, volatility, use_heston=True)
        mc_prob_profit_heston = np.mean(mc_prices_heston[-1] > short_put['strike']) if len(mc_prices_heston) > 1 else 0

        # Bayesian Network Prediction
        bayesian_prob = bayesian_network_prediction(probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, mc_prob_profit_heston)

        if bayesian_prob < min_prob_success:
            return []

        return [(exp, short_put['strike'], long_put['strike'], credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, mc_prob_profit_heston, bayesian_prob, pricing_state)]
    return []

def bayesian_network_prediction(prob_success, mc_prob_no_dte, mc_prob_with_dte, mc_prob_heston):
    model = BayesianNetwork([('ProbSuccess', 'FinalProb'), ('MCNoDTE', 'FinalProb'), ('MCWithDTE', 'FinalProb'), ('MCHeston', 'FinalProb')])

    cpd_prob_success = TabularCPD(variable='ProbSuccess', variable_card=2, values=[[1 - prob_success], [prob_success]])
    cpd_mc_no_dte = TabularCPD(variable='MCNoDTE', variable_card=2, values=[[1 - mc_prob_no_dte], [mc_prob_no_dte]])
    cpd_mc_with_dte = TabularCPD(variable='MCWithDTE', variable_card=2, values=[[1 - mc_prob_with_dte], [mc_prob_with_dte]])
    cpd_mc_heston = TabularCPD(variable='MCHeston', variable_card=2, values=[[1 - mc_prob_heston], [mc_prob_heston]])

    # Define helper function to calculate joint probability for FinalProb
    def calculate_joint_prob(prob_success, mc_prob_no_dte, mc_prob_with_dte, mc_prob_heston):
        return (prob_success + mc_prob_no_dte + mc_prob_with_dte + mc_prob_heston) / 4

    # Generate CPD values dynamically
    values = []
    for ps in [0, 1]:
        for mcnd in [0, 1]:
            for mcwd in [0, 1]:
                for mch in [0, 1]:
                    if ps + mcnd + mcwd + mch == 0:
                        prob = 0.05  # Low probability when all inputs are negative
                    else:
                        prob = calculate_joint_prob(
                            prob_success if ps else 1 - prob_success,
                            mc_prob_no_dte if mcnd else 1 - mc_prob_no_dte,
                            mc_prob_with_dte if mcwd else 1 - mc_prob_with_dte,
                            mc_prob_heston if mch else 1 - mc_prob_heston
                        )
                    values.append([1 - prob, prob])

    values = np.array(values).T

    cpd_final_prob = TabularCPD(
        variable='FinalProb',
        variable_card=2,
        values=values,
        evidence=['ProbSuccess', 'MCNoDTE', 'MCWithDTE', 'MCHeston'],
        evidence_card=[2, 2, 2, 2]
    )

    model.add_cpds(cpd_prob_success, cpd_mc_no_dte, cpd_mc_with_dte, cpd_mc_heston, cpd_final_prob)

    # Inference
    infer = VariableElimination(model)
    query = infer.query(variables=['FinalProb'], evidence={
        'ProbSuccess': 1 if prob_success > 0.5 else 0,
        'MCNoDTE': 1 if mc_prob_no_dte > 0.5 else 0,
        'MCWithDTE': 1 if mc_prob_with_dte > 0.5 else 0,
        'MCHeston': 1 if mc_prob_heston > 0.5 else 0
    })

    # Return the continuous probability of success
    return query.values[1]

def process_bull_put_spread(symbol, exp, put_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, api_token, cache, min_prob_success):
    results = []
    put_chain = put_chain.sort_values(by='strike')
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_spread, i, j, put_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, api_token, symbol, exp, cache, min_prob_success) for i in range(len(put_chain)) for j in range(i + 1, len(put_chain))]
        total_spreads = len(futures)
        completed_spreads = 0

        for future in as_completed(futures):
            results.extend(future.result())
            completed_spreads += 1
            if completed_spreads % 100 == 0 or completed_spreads == total_spreads:
                logger.info(f"Processed {completed_spreads}/{total_spreads} bull put spreads for {symbol} with expiration {exp}")

    return results

def process_single_condor(i, j, k, l, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, exp, min_prob_success):
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

        # Monte Carlo Simulation with and without DTE adjustment
        mc_prices_no_dte = monte_carlo_simulation(underlying_price, int(time_to_expiration), simulations, volatility, use_t_dist=True)
        mc_prob_profit_put_no_dte = np.mean(mc_prices_no_dte[-1] > short_put['strike'])
        mc_prob_profit_call_no_dte = np.mean(mc_prices_no_dte[-1] < short_call['strike'])
        mc_prob_profit_no_dte = (mc_prob_profit_put_no_dte + mc_prob_profit_call_no_dte) / 2

        adjusted_volatility = get_historical_volatility_adjusted(symbol, api_token, time_to_expiration, cache)
        mc_prices_with_dte = monte_carlo_simulation(underlying_price, int(time_to_expiration), simulations, adjusted_volatility, use_t_dist=True)
        mc_prob_profit_put_with_dte = np.mean(mc_prices_with_dte[-1] > short_put['strike'])
        mc_prob_profit_call_with_dte = np.mean(mc_prices_with_dte[-1] < short_call['strike'])
        mc_prob_profit_with_dte = (mc_prob_profit_put_with_dte + mc_prob_profit_call_with_dte) / 2

        # Heston Model Simulations
        mc_prices_heston = monte_carlo_simulation(underlying_price, int(time_to_expiration), simulations, volatility, use_heston=True)
        mc_prob_profit_put_heston = np.mean(mc_prices_heston[-1] > short_put['strike'])
        mc_prob_profit_call_heston = np.mean(mc_prices_heston[-1] < short_call['strike'])
        mc_prob_profit_heston = (mc_prob_profit_put_heston + mc_prob_profit_call_heston) / 2

        # Bayesian Network Prediction
        bayesian_prob = bayesian_network_prediction(probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, mc_prob_profit_heston)

        if bayesian_prob < min_prob_success:
            return []

        return [(exp, short_put['strike'], long_put['strike'], short_call['strike'], long_call['strike'], credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, mc_prob_profit_heston, bayesian_prob, pricing_state_put, pricing_state_call)]
    return []

def process_iron_condor(symbol, exp, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, api_token, cache, min_prob_success):
    results = []
    put_chain = put_chain.sort_values(by='strike')
    call_chain = call_chain.sort_values(by='strike')

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_condor, i, j, k, l, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, exp, min_prob_success)
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

def find_bull_put_spreads(puts, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility, risk_free_rate, api_token, cache, min_prob_success):
    logger.info(f"Finding bull put spreads for underlying price {underlying_price} using batch size {batch_size}")
    bull_put_spreads = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures_iterator = (
            executor.submit(process_bull_put_spread, symbol, exp, put_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, api_token, cache, min_prob_success)
            for symbol, exp, put_chain in puts
        )
        batches_processed = 0
        for batch in batch_futures(futures_iterator, batch_size):
            for future in as_completed(batch):
                bull_put_spreads.extend(future.result())
            batches_processed += 1
            logger.info(f"Processed batch {batches_processed}")

    return bull_put_spreads

def find_iron_condors(puts, calls, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility, risk_free_rate, api_token, cache, min_prob_success):
    logger.info(f"Finding iron condors for underlying price {underlying_price} using batch size {batch_size}")
    iron_condors = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures_iterator = (
            executor.submit(process_iron_condor, symbol, exp, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility, risk_free_rate, api_token, cache, min_prob_success)
            for (symbol, exp, put_chain), (symbol_c, exp_c, call_chain) in zip(puts, calls)
            if symbol == symbol_c and exp == exp_c
        )
        batches_processed = 0
        for batch in batch_futures(futures_iterator, batch_size):
            for future in as_completed(batch):
                iron_condors.extend(future.result())
            batches_processed += 1
            logger.info(f"Processed batch {batches_processed} for iron condors for symbol {symbol} with expiration {exp}")

    return iron_condors

def find_best_spreads(symbols, puts, calls, top_n, min_ror, max_strike_dist, batch_size, simulations, risk_free_rate, api_token, find_ic, min_prob_success):
    combined_spreads = []

    with Manager() as manager:
        cache = manager.dict()
        
        for symbol in symbols:
            underlying_price = get_stock_price(symbol, api_token)
            volatility = get_historical_volatility(symbol, api_token, cache)
            
            symbol_puts = [put for put in puts if put[0] == symbol]
            symbol_calls = [call for call in calls if call[0] == symbol]

            bull_put_spreads = find_bull_put_spreads(symbol_puts, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility, risk_free_rate, api_token, cache, min_prob_success)
            for spread in bull_put_spreads:
                combined_spreads.append(list((*spread, 'bull_put', symbol)))  # Convert tuple to list and add symbol

            if find_ic:
                iron_condors = find_iron_condors(symbol_puts, symbol_calls, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility, risk_free_rate, api_token, cache, min_prob_success)
                for spread in iron_condors:
                    combined_spreads.append(list((*spread, 'iron_condor', symbol)))  # Convert tuple to list and add symbol

    # Filter out spreads that don't meet the minimum return on risk
    filtered_spreads = [spread for spread in combined_spreads if spread[5] >= min_ror]

    # Calculate average probability using all Monte Carlo probabilities
    for spread in filtered_spreads:
        spread.append((spread[7] + spread[8] + spread[9] + spread[10]) / 4)  # Adding average_probability using all MC probabilities

    # Sort by average probability, prioritizing the highest probabilities
    filtered_spreads.sort(key=lambda x: x[-1], reverse=True)

    spread_data = []
    for spread in filtered_spreads[:top_n]:
        if spread[-3] == 'bull_put':
            exp, short_strike, long_strike, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, mc_prob_profit_heston, bayesian_prob, pricing_state, spread_type, symbol, average_probability = spread
            spread_data.append([symbol, spread_type, exp, short_strike, long_strike, None, None, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, mc_prob_profit_heston, bayesian_prob, pricing_state, None, average_probability])
        elif spread[-3] == 'iron_condor':
            exp, short_put_strike, long_put_strike, short_call_strike, long_call_strike, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, mc_prob_profit_heston, bayesian_prob, pricing_state_put, pricing_state_call, spread_type, symbol, average_probability = spread
            spread_data.append([symbol, spread_type, exp, short_put_strike, long_put_strike, short_call_strike, long_call_strike, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit_no_dte, mc_prob_profit_with_dte, mc_prob_profit_heston, bayesian_prob, pricing_state_put, pricing_state_call, average_probability])

    return spread_data

def backtest_portfolio(symbols, puts, calls, start_date, end_date, initial_cash, stop_loss_pct, max_positions, min_profit_pct, min_prob_success, min_ror, max_strike_dist, batch_size, simulations, risk_free_rate, api_token, find_ic, margin_multiplier):
    initial_cash *= (1 + margin_multiplier)
    cash = initial_cash
    positions = []
    equity_curve = []
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    for current_date in dates:
        logger.info(f"Processing date: {current_date.strftime('%Y-%m-%d')} for backtesting portfolio including {len(positions)} positions")
        
        # Check for expiring positions and close them
        positions = [pos for pos in positions if datetime.datetime.strptime(pos['expiration'], '%Y-%m-%d') > current_date]
        
        # Evaluate current portfolio
        current_equity = cash + sum(pos['credit'] for pos in positions)
        equity_curve.append((current_date, current_equity))
        
        # Check for stop loss and minimum profit conditions
        positions_to_remove = []
        for pos in positions:
            if pos['max_loss'] / initial_cash >= stop_loss_pct or pos['credit'] / pos['max_loss'] >= min_profit_pct:
                cash += pos['credit']
                positions_to_remove.append(pos)
        
        for pos in positions_to_remove:
            positions.remove(pos)
        
        # Fetch new best spreads
        best_spreads = find_best_spreads(symbols, puts, calls, max_positions - len(positions), min_ror, max_strike_dist, batch_size, simulations, risk_free_rate, api_token, find_ic, min_prob_success)
        
        # Allocate new positions
        for spread in best_spreads:
            if len(positions) < max_positions and cash >= spread[7]:
                positions.append({
                    'expiration': spread[2],
                    'short_strike': spread[3],
                    'long_strike': spread[4],
                    'credit': spread[7],
                    'max_loss': spread[8],
                    'return_on_risk': spread[9],
                    'probability_of_success': spread[10]
                })
                cash -= spread[7]
    
    # Calculate final metrics
    final_equity = cash + sum(pos['credit'] for pos in positions)
    returns = (final_equity - initial_cash) / initial_cash
    equity_curve_df = pd.DataFrame(equity_curve, columns=['Date', 'Equity'])
    equity_curve_df.set_index('Date', inplace=True)
    
    # Calculate Sharpe ratio
    daily_returns = equity_curve_df['Equity'].pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    
    return equity_curve_df, sharpe_ratio, returns

def plot_backtest_results(equity_curve_df, sharpe_ratio, returns):
    plt.figure(figsize=(12, 8))
    plt.plot(equity_curve_df.index, equity_curve_df['Equity'], label='Equity Curve')
    plt.title(f'Backtest Results - Final Return: {returns:.2f}, Sharpe Ratio: {sharpe_ratio:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    startTime = datetime.datetime.now()
    argparser = argparse.ArgumentParser(description="Find and rank option spreads")

    # General arguments
    argparser.add_argument('-symbols', '-s', type=str, required=True, help='Comma-separated list of stock symbols to fetch data for')
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
    argparser.add_argument('--plot', action='store_true', help='Show probability of profit plot')
    argparser.add_argument('--backtesting', action='store_true', help='Enable backtesting')
    argparser.add_argument('-min_prob_success', '-ps', type=float, default=0.5, help='Minimum probability of success based on Bayesian probability')

    # Backtesting-specific arguments
    backtesting_args = argparse.ArgumentParser(add_help=False)
    backtesting_args.add_argument('-start_cash', '-c', type=float, required=True, help='Starting cash amount')
    backtesting_args.add_argument('-stop_loss_pct', '-sl', type=float, required=True, help='Stop loss percentage of starting cash')
    backtesting_args.add_argument('-max_positions', '-mp', type=int, required=True, help='Maximum number of positions at once')
    backtesting_args.add_argument('-min_profit_pct', '-mpf', type=float, required=True, help='Minimum profit percentage to close the position')
    backtesting_args.add_argument('-years', '-y', type=int, required=True, help='Number of years to backtest')
    backtesting_args.add_argument('-margin', '-mg', type=float, required=True, help='Margin multiplier as a percentage of starting cash')

    # Parse the known arguments first
    args, unknown = argparser.parse_known_args()

    if args.backtesting:
        # Parse the backtesting-specific arguments if backtesting is enabled
        argparser = argparse.ArgumentParser(parents=[argparser, backtesting_args])
        args = argparser.parse_args()

    symbols = args.symbols.split(',')
    start_date = datetime.datetime.now() - datetime.timedelta(days=365 * args.years) if args.backtesting else None
    end_date = datetime.datetime.now()

    puts, calls = get_stock_data(symbols, args.mindte, args.maxdte, args.api_token)

    logger.info(f"Symbols: {symbols}")

    if args.backtesting:
        equity_curve_df, sharpe_ratio, returns = backtest_portfolio(
            symbols, puts, calls, start_date, end_date, args.start_cash, args.stop_loss_pct, args.max_positions, args.min_profit_pct,
            args.min_prob_success, args.min_ror, args.max_strike_dist, args.batch_size, args.simulations, args.risk_free_rate, args.api_token, args.include_iron_condors, args.margin
        )

        logger.info(f"Final Returns: {returns:.2f}, Sharpe Ratio: {sharpe_ratio:.2f}")

        if args.plot:
            plot_backtest_results(equity_curve_df, sharpe_ratio, returns)
    else:
        best_spreads = find_best_spreads(symbols, puts, calls, args.top_n, args.min_ror, args.max_strike_dist, args.batch_size, args.simulations, args.risk_free_rate, args.api_token, args.include_iron_condors, args.min_prob_success)

        logger.info(f"Top {args.top_n} spreads:")
        for spread in best_spreads:
            if spread[1] == 'bull_put':
                logger.info(
                    f"Symbol: {spread[0]}, Type: {spread[1]}, Expiration: {spread[2]}\n"
                    f"Short Put Strike: {spread[3]}, Long Put Strike: {spread[4]}\n"
                    f"Credit: ${spread[7]:.2f}, Max Loss: ${spread[8]:.2f}, RoR: {spread[9] * 100:.2f}%\n"
                    f"Probability of Success: {spread[10] * 100:.2f}%, MC Profit No DTE: {spread[11] * 100:.2f}%\n"
                    f"MC Profit With DTE: {spread[12] * 100:.2f}%, MC Profit Heston: {spread[13] * 100:.2f}%\n"
                    f"Bayesian Probability: {spread[14] * 100:.2f}%, Pricing State: {spread[15]}\n"
                    f"Average Probability: {spread[-1] * 100:.2f}%\n"
                )
            elif spread[1] == 'iron_condor':
                logger.info(
                    f"Symbol: {spread[0]}, Type: {spread[1]}, Expiration: {spread[2]}\n"
                    f"Short Put Strike: {spread[3]}, Long Put Strike: {spread[4]}\n"
                    f"Short Call Strike: {spread[5]}, Long Call Strike: {spread[6]}\n"
                    f"Credit: ${spread[7]:.2f}, Max Loss: ${spread[8]:.2f}, RoR: {spread[9] * 100:.2f}%\n"
                    f"Probability of Success: {spread[10] * 100:.2f}%, MC Profit No DTE: {spread[11] * 100:.2f}%\n"
                    f"MC Profit With DTE: {spread[12] * 100:.2f}%, MC Profit Heston: {spread[13] * 100:.2f}%\n"
                    f"Bayesian Probability: {spread[14] * 100:.2f}%, Pricing State Put: {spread[15]}\n"
                    f"Pricing State Call: {spread[16]}, Average Probability: {spread[-1] * 100:.2f}%\n"
                )
            print("---")

        pd.DataFrame(best_spreads, columns=[
            'Symbol', 'Spread Type', 'Expiration', 'Short Put Strike', 'Long Put Strike', 'Short Call Strike', 'Long Call Strike',
            'Credit', 'Max Loss', 'Return on Risk', 'Probability of Success', 'MC Profit No DTE', 'MC Profit With DTE',
            'MC Profit Heston', 'Bayesian Probability', 'Pricing State Put', 'Pricing State Call', 'Average Probability'
        ]).to_csv(args.output, index=False)

    endTime = datetime.datetime.now()
    logger.info(f"Time taken: {endTime - startTime}")
