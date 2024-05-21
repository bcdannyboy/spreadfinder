import argparse
import datetime
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import islice
import os

TRADIER_API_URL = "https://api.tradier.com/v1"

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
    volatility = np.std(log_returns) * np.sqrt(days)  # annualized volatility
    return volatility

def get_stock_data(symbol, mindte, maxdte, api_token):
    print(f"Fetching stock data for {symbol} with min DTE {mindte} and max DTE {maxdte}")
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

    print(f"Found {len(puts)} put chains and {len(calls)} call chains for {symbol} with min DTE {mindte} and max DTE {maxdte}")
    return puts, calls

def monte_carlo_simulation(current_price, days, simulations, volatility):
    dt = 1 / 252  # daily time step
    prices = np.zeros((days + 1, simulations))
    prices[0] = current_price

    for t in range(1, days + 1):
        z = np.random.standard_normal(simulations)
        prices[t] = prices[t - 1] * np.exp((0 - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * z)

    return prices

def black_scholes_price(option_type, S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def process_bull_put_spread(exp, put_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility):
    risk_free_rate = 0.01  # Assuming a risk-free rate of 1%

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

            theoretical_price = black_scholes_price('put', underlying_price, short_put['strike'], time_to_expiration, risk_free_rate, iv)
            if theoretical_price > short_put['bid']:
                underpriced = True
            else:
                underpriced = False

            d1 = (np.log(underlying_price / short_put['strike']) + (0.5 * iv**2) * time_to_expiration) / (iv * np.sqrt(time_to_expiration))
            probability_of_success = norm.cdf(d1)

            # Monte Carlo Simulation
            mc_prices = monte_carlo_simulation(underlying_price, int(time_to_expiration * 252), simulations, iv)
            mc_prob_profit = np.mean(mc_prices[-1] > short_put['strike'])

            return [(exp, short_put['strike'], long_put['strike'], credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit, underpriced)]
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
                print(f"\rProcessed {completed_spreads}/{total_spreads} bull put spreads", end='', flush=True)

    print()
    return results

def batch_futures(iterator, batch_size):
    """Helper function to batch futures."""
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def find_bull_put_spreads(puts, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility):
    print(f"Finding bull put spreads for underlying price {underlying_price} using batch size {batch_size}")
    bull_put_spreads = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures_iterator = (
            executor.submit(process_bull_put_spread, exp, put_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility)
            for exp, put_chain in puts
        )
        batches_processed = 0
        for batch in batch_futures(futures_iterator, batch_size):
            for future in as_completed(batch):
                bull_put_spreads.extend(future.result())
            batches_processed += 1
            print(f"\rProcessed batch {batches_processed}", end='', flush=True)

    print()
    return bull_put_spreads

def process_iron_condor(exp, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility):
    risk_free_rate = 0.01  # Assuming a risk-free rate of 1%

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

            theoretical_price_put = black_scholes_price('put', underlying_price, short_put['strike'], time_to_expiration, risk_free_rate, iv_put)
            theoretical_price_call = black_scholes_price('call', underlying_price, short_call['strike'], time_to_expiration, risk_free_rate, iv_call)

            if theoretical_price_put > short_put['bid']:
                underpriced_put = True
            else:
                underpriced_put = False

            if theoretical_price_call > short_call['bid']:
                underpriced_call = True
            else:
                underpriced_call = False

            d1_put = (np.log(underlying_price / short_put['strike']) + (0.5 * iv_put**2) * time_to_expiration) / (iv_put * np.sqrt(time_to_expiration))
            d1_call = (np.log(underlying_price / short_call['strike']) + (0.5 * iv_call**2) * time_to_expiration) / (iv_call * np.sqrt(time_to_expiration))
            probability_of_success_put = norm.cdf(d1_put)
            probability_of_success_call = 1 - norm.cdf(d1_call)
            probability_of_success = (probability_of_success_put + probability_of_success_call) / 2

            # Monte Carlo Simulation
            mc_prices = monte_carlo_simulation(underlying_price, int(time_to_expiration * 252), simulations, iv_put)
            mc_prob_profit_put = np.mean(mc_prices[-1] > short_put['strike'])
            mc_prob_profit_call = np.mean(mc_prices[-1] < short_call['strike'])
            mc_prob_profit = (mc_prob_profit_put + mc_prob_profit_call) / 2

            return [(exp, short_put['strike'], long_put['strike'], short_call['strike'], long_call['strike'], credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit, underpriced_put, underpriced_call)]
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
                print(f"\rProcessed {completed_condors}/{total_condors} iron condors", end='', flush=True)

    print()
    return results

def find_iron_condors(puts, calls, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility):
    print(f"Finding iron condors for underlying price {underlying_price} using batch size {batch_size}")
    iron_condors = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures_iterator = (
            executor.submit(process_iron_condor, exp, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, simulations, volatility)
            for (exp, put_chain), (exp_c, call_chain) in zip(puts, calls)
            if exp == exp_c
        )
        batches_processed = 0
        for batch in batch_futures(futures_iterator, batch_size):
            for future in as_completed(batch):
                iron_condors.extend(future.result())
            batches_processed += 1
            print(f"\rProcessed batch {batches_processed}", end='', flush=True)

    print()
    return iron_condors

def find_best_spreads(puts, calls, underlying_price, top_n, min_ror, max_strike_dist, batch_size, simulations, volatility, include_iron_condors):
    bull_put_spreads = find_bull_put_spreads(puts, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility)
    combined_spreads = []

    for spread in bull_put_spreads:
        combined_spreads.append((*spread, 'bull_put'))

    if include_iron_condors:
        iron_condors = find_iron_condors(puts, calls, underlying_price, min_ror, max_strike_dist, batch_size, simulations, volatility)
        for spread in iron_condors:
            combined_spreads.append((*spread, 'iron_condor'))

    combined_spreads.sort(key=lambda x: (x[-3], x[-2]), reverse=True)

    spread_data = []
    for spread in combined_spreads:
        if spread[-1] == 'bull_put':
            exp, short_strike, long_strike, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit, underpriced, spread_type = spread
            spread_data.append([spread_type, exp, short_strike, long_strike, None, None, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit, underpriced, None])
        elif spread[-1] == 'iron_condor':
            exp, short_put_strike, long_put_strike, short_call_strike, long_call_strike, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit, underpriced_put, underpriced_call, spread_type = spread
            spread_data.append([spread_type, exp, short_put_strike, long_put_strike, short_call_strike, long_call_strike, credit, max_loss, return_on_risk, probability_of_success, mc_prob_profit, underpriced_put, underpriced_call])

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

    args = argparser.parse_args()
    
    underlying_price = get_stock_price(args.symbol, args.api_token)
    volatility = get_historical_volatility(args.symbol, args.api_token)
    puts, calls = get_stock_data(args.symbol, args.mindte, args.maxdte, args.api_token)
    
    print(f"Underlying price: {underlying_price}")
    print(f"Historical volatility: {volatility}")

    best_spreads = find_best_spreads(puts, calls, underlying_price, args.top_n, args.min_ror, args.max_strike_dist, args.batch_size, args.simulations, volatility, args.include_iron_condors)
    
    print("Best Spreads:")
    df = pd.DataFrame(best_spreads, columns=['Type', 'Expiration', 'Short Put Strike', 'Long Put Strike', 'Short Call Strike', 'Long Call Strike', 'Credit', 'Max Loss', 'Return on Risk', 'Probability of Success', 'MC Probability of Profit', 'Underpriced Put', 'Underpriced Call'])
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

    for spread in best_spreads:
        if spread[0] == 'bull_put':
            print(f"Bull Put Spread:\n\tExpiration: {spread[1]}\n\tShort Strike: {spread[2]}\n\tLong Strike: {spread[3]}\n\tCredit: {spread[6]}\n\tMax Loss: {spread[7]}\n\tReturn on Risk: {spread[8]*100:.2f}%\n\tProbability of Success: {spread[9]*100:.2f}%\n\tMC Probability of Profit: {spread[10]*100:.2f}%\n\tUnderpriced: {spread[11]}")
        elif spread[0] == 'iron_condor':
            print(f"Iron Condor:\n\tExpiration: {spread[1]}\n\tShort Put Strike: {spread[2]}\n\tLong Put Strike: {spread[3]}\n\tShort Call Strike: {spread[4]}\n\tLong Call Strike: {spread[5]}\n\tCredit: {spread[6]}\n\tMax Loss: {spread[7]}\n\tReturn on Risk: {spread[8]*100:.2f}%\n\tProbability of Success: {spread[9]*100:.2f}%\n\tMC Probability of Profit: {spread[10]*100:.2f}%\n\tUnderpriced Put: {spread[11]}\n\tUnderpriced Call: {spread[12]}")

    endTime = datetime.datetime.now()
    print(f"Time taken: {endTime - startTime}")
